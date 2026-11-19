// LlamaZip, a port of the LlamaZip class in llama_zip.py to txiki.js.
//
//   const zip = new LlamaZip(modelPath, { nCtx, nGpuLayers, useMlock, nThreads, nBatch, verbose });
//   const compressed = await zip.compress(bytes, windowOverlap);   // Uint8Array
//   const original = await zip.decompress(compressed, windowOverlap); // Uint8Array
//
// compress and decompress are async so the event loop runs now and then, which
// is what lets a SIGINT listener and the progress bar work while the model is
// busy. The format is the one llama_zip.py writes.
//
// Determinism. The decoder replays the encoder's computation, so both sides
// must produce bit-identical logits. They do when they use the same model,
// the same libllama build, the same thread count and the same evaluation
// chunks. This port always evaluates a window the way llama-cpp-python's
// generate() does: reset, evaluate [BOS] + overlap tokens in chunks of at most
// nBatch, then one token per llama_decode call. Nothing about this is stored in the
// compressed data, so compress and decompress need the same nThreads and
// nBatch (defaults below) as well as the same window overlap.
//
// Differences from llama-cpp-python that change the bits: it reuses the KV
// cache when a window prefix matches the previous one, it runs generation and
// prompt evaluation with different thread counts, and it computes the
// log-softmax in float32. See computeCdf for the numerics.
import { LlamaModel } from './llama.js';
import { Encoder, Decoder } from './arithmetic.js';
import { PUA_START, bytesToUtf8, utf8ToBytes } from './utf8.js';
import { defaultIo } from './io.js';
import { StepBuilder } from './trace.js';

export { PUA_START };

export const FREQ_SCALE_FACTOR = 2 ** 32;

// Both sides must use the same value. Fixed instead of the machine's core count so a file
// compressed on one machine decodes on another with the same libllama.
export const DEFAULT_N_THREADS = 4;
export const DEFAULT_N_BATCH = 512;

const LOADING_MESSAGE = 'Loading model...';

// The decoder reads 64 bits ahead of the encoder's output, so a valid stream
// is consumed after at most 64 bits beyond its length.
const DECODER_LOOKAHEAD_BITS = 64;

// numpy.round: halves go to the even neighbour. Math.round sends them up.
export const roundHalfEven = value => {
    const floor = Math.floor(value);
    const diff = value - floor;
    if (diff < 0.5) return floor;
    if (diff > 0.5) return floor + 1;
    return floor % 2 === 0 ? floor : floor + 1;
};

// Sum of exp(logits[i] - max) over a range, in numpy's pairwise order (eight running sums
// per block of up to 128, blocks combined by halving), so the total matches np.sum of the
// exponentials bit for bit. The exponentials are computed inside the sum instead of in a
// separate pass, and the block sums are unrolled by hand, because this runs once per
// vocabulary entry for every token and the interpreter is slow.
const pairwiseExpSum = (logits, max, start, count) => {
    if (count < 8) {
        let total = 0;
        for (let i = 0; i < count; i++) total += Math.exp(logits[start + i] - max);
        return total;
    }
    if (count <= 128) {
        let r0 = Math.exp(logits[start] - max);
        let r1 = Math.exp(logits[start + 1] - max);
        let r2 = Math.exp(logits[start + 2] - max);
        let r3 = Math.exp(logits[start + 3] - max);
        let r4 = Math.exp(logits[start + 4] - max);
        let r5 = Math.exp(logits[start + 5] - max);
        let r6 = Math.exp(logits[start + 6] - max);
        let r7 = Math.exp(logits[start + 7] - max);
        let i = 8;
        const blockEnd = count - (count % 8);
        for (; i < blockEnd; i += 8) {
            const k = start + i;
            r0 += Math.exp(logits[k] - max);
            r1 += Math.exp(logits[k + 1] - max);
            r2 += Math.exp(logits[k + 2] - max);
            r3 += Math.exp(logits[k + 3] - max);
            r4 += Math.exp(logits[k + 4] - max);
            r5 += Math.exp(logits[k + 5] - max);
            r6 += Math.exp(logits[k + 6] - max);
            r7 += Math.exp(logits[k + 7] - max);
        }
        let total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < count; i++) total += Math.exp(logits[start + i] - max);
        return total;
    }
    let half = Math.floor(count / 2);
    half -= half % 8;
    return pairwiseExpSum(logits, max, start, half) + pairwiseExpSum(logits, max, start + half, count - half);
};

// Largest finite logit, 0 if there is none. Math.max over blocks of values is native and fast
// (an engine accepts only about 65000 arguments per call), and a loop takes over if a NaN is present.
const APPLY_CHUNK = 32768;

const maxLogit = logits => {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i += APPLY_CHUNK) {
        const part = logits.subarray ? logits.subarray(i, i + APPLY_CHUNK) : logits.slice(i, i + APPLY_CHUNK);
        max = Math.max(max, Math.max.apply(null, part));
    }
    if (Number.isNaN(max)) {
        max = -Infinity;
        for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    }
    return Number.isFinite(max) ? max : 0;
};

// A token whose log probability is below this has a scaled probability under 1.5, which
// rounds to at most 1, so the floor of 1 applies without evaluating exp. exp(-22) * 2^32 is 1.2.
const FLOOR_LOG_PROBABILITY = -22;

// Cumulative frequencies for a coder, as a Float64Array of integers.
//
// Numerics. llama-cpp-python's logits_to_logprobs works in float32. Float32 exp,
// log and sum in numpy depend on the CPU's SIMD kernels and cannot be reproduced
// in JavaScript, so this port does the whole softmax in float64: float32 logits
// are widened, the maximum is subtracted, and the log-softmax follows the
// same formulas as the float64 reference in tests/vectors/compute_cdf.json. The
// scaled probabilities are rounded half to even (roundHalfEven, inlined below), floored
// at 1, and summed. Totals stay near 2^32 + vocab size, far below 2^53, so doubles hold
// every value exactly. The result is not guaranteed to equal the Python tool's float32
// result, so streams are only guaranteed to decode within this implementation.
export const computeCdf = logits => {
    const n = logits.length;
    const max = maxLogit(logits);
    const logSum = Math.log(pairwiseExpSum(logits, max, 0, n));
    const cutoff = max + logSum + FLOOR_LOG_PROBABILITY;
    const cdf = new Float64Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
        const logit = logits[i];
        let frequency = 1;
        if (logit >= cutoff) {
            const scaled = FREQ_SCALE_FACTOR * Math.exp((logit - max) - logSum);
            // Math.round sends ties up, numpy.round to the even neighbour. Below 2^52 the tie test is exact.
            frequency = Math.round(scaled);
            if (frequency - scaled === 0.5 && frequency % 2 === 1) frequency -= 1;
            if (!(frequency >= 1)) frequency = 1;
        }
        total += frequency;
        cdf[i] = total;
    }
    return cdf;
};

// A progress line on stderr like tqdm's, removed when closed (leave=False).
class ProgressBar {
    constructor(sink, { total, desc, unit }) {
        this.sink = sink;
        this.total = total;
        this.desc = desc;
        this.unit = unit;
        this.count = 0;
        this.start = Date.now();
        this.lastRender = 0;
        this.width = 0;
    }

    update() {
        this.count++;
        const now = Date.now();
        if (now - this.lastRender >= 1000 / 30) this.render(now);
    }

    render(now) {
        this.lastRender = now;
        const elapsed = (now - this.start) / 1000;
        const fraction = this.total ? this.count / this.total : 0;
        const filled = Math.floor(fraction * 10);
        const bar = '#'.repeat(filled) + ' '.repeat(10 - filled);
        const rate = elapsed > 0 ? this.count / elapsed : 0;
        const clock = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(Math.floor(elapsed % 60)).padStart(2, '0')}`;
        const text = `${this.desc}: ${String(Math.floor(fraction * 100)).padStart(3)}%|${bar}| ${this.count}/${this.total} [${clock}, ${rate.toFixed(1)}${this.unit}/s]`;
        this.width = Math.max(this.width, text.length);
        this.sink.write('\r' + text.padEnd(this.width));
    }

    close() {
        if (this.lastRender) this.sink.write('\r' + ' '.repeat(this.width) + '\r');
    }
}

const yieldToEventLoop = () => new Promise(resolve => setTimeout(resolve, 1));

export class LlamaZip {
    // Options: nCtx (0 = model maximum), nGpuLayers (-1 = all), useMlock, nThreads, nBatch,
    // verbose (progress, loading message and streamed output on stderr/stdout),
    // createModel (factory taking the LlamaModel options, for tests), io ({ out, err } sinks).
    constructor(modelPath, {
        nCtx = 0,
        nGpuLayers = -1,
        useMlock = false,
        nThreads = DEFAULT_N_THREADS,
        nBatch = DEFAULT_N_BATCH,
        verbose = false,
        createModel = options => new LlamaModel(options),
        io = null,
        loadingMessage = true,
    } = {}) {
        this.verbose = verbose;
        this.io = io;
        this.lastYield = Date.now();
        this.busy = false;
        const showLoading = verbose && loadingMessage;
        if (showLoading) this.#sinks().err.write(LOADING_MESSAGE);
        this.model = createModel({ path: modelPath, nCtx, nGpuLayers, useMlock, nThreads, nBatch });
        if (showLoading) this.#sinks().err.write('\r' + ' '.repeat(LOADING_MESSAGE.length) + '\r');
    }

    // Like the constructor, but lets the loading message reach the terminal before the
    // synchronous model load blocks the event loop.
    static async load(modelPath, options = {}) {
        const sinks = options.io ?? defaultIo();
        const showLoading = options.verbose && options.loadingMessage !== false;
        if (showLoading) {
            sinks.err.write(LOADING_MESSAGE);
            await sinks.err.flush();
            await yieldToEventLoop();
        }
        const zip = new LlamaZip(modelPath, { ...options, io: sinks, loadingMessage: false });
        if (showLoading) sinks.err.write('\r' + ' '.repeat(LOADING_MESSAGE.length) + '\r');
        return zip;
    }

    #sinks() {
        return this.io ??= defaultIo();
    }

    close() {
        this.model.close();
    }

    computeCdf(logits) {
        return computeCdf(logits);
    }

    #checkOverlap(windowOverlap) {
        if (!Number.isInteger(windowOverlap) || windowOverlap < 0 || windowOverlap >= this.model.nCtx) {
            throw new RangeError(`windowOverlap must be an integer in the range [0, ${this.model.nCtx - 1}], got ${windowOverlap}`);
        }
    }

    // Lets the event loop run about every 50 ms.
    async #maybeYield() {
        const now = Date.now();
        if (now - this.lastYield < 50) return;
        await yieldToEventLoop();
        this.lastYield = Date.now();
    }

    // Start of a window: clear the context and evaluate BOS plus the overlap tokens in one go, as generate() does.
    // Returns the number of tokens now in the context.
    #startWindow(overlapTokens) {
        this.model.reset();
        const prefix = [this.model.tokenBos(), ...overlapTokens];
        this.model.eval(prefix);
        return prefix.length;
    }

    // Abandoned or failed call: the context holds a half-finished window, so clear it. The model
    // may be closed already, which leaves no state to clear.
    #clearContext() {
        try {
            this.model.reset();
        }
        catch {
            // Nothing to clear.
        }
    }

    // Marks the model as in use until the returned function is called, so two calls cannot
    // interleave their evaluations on the one context.
    #claim() {
        if (this.busy) throw new Error('LlamaZip is already compressing or decompressing, wait for the other call to finish');
        this.busy = true;
        return () => {
            this.busy = false;
        };
    }

    // Aborting `signal` stops early: the rest of the input is dropped and the stream is finished
    // with EOS, so the output still decodes to the part that was compressed.
    //
    // onStep(step) is called after each coded token (the EOS included) with the object that
    // js/trace.js describes. If it returns a promise, compress waits for it, so a caller can slow
    // the work down to the pace of a display. topK sets the number of candidates per step.
    // Observers never change the compressed bytes.
    async compress(uncompressed, windowOverlap = 0, { signal = null, onStep = null, topK, outputBitsCap } = {}) {
        const steps = this.#compressSteps(uncompressed, windowOverlap, { signal, observe: onStep !== null, topK, outputBitsCap });
        try {
            for (;;) {
                const next = await steps.next();
                if (next.done) return next.value.compressed;
                await onStep(next.value);
            }
        }
        finally {
            await steps.return();
        }
    }

    // Like compress, as an async generator that yields each step and returns
    // { compressed, totalBits, tokens }. totalBits is the sum of the steps' bits (the ideal code
    // length), and the compressed size is that rounded up plus the few bits the coder's final
    // interval needs. Calling return() on the generator early stops the work and resets the model.
    async *trace(uncompressed, windowOverlap = 0, { signal = null, topK, outputBitsCap } = {}) {
        const steps = this.#compressSteps(uncompressed, windowOverlap, { signal, observe: true, topK, outputBitsCap });
        try {
            for (;;) {
                const next = await steps.next();
                if (next.done) return next.value;
                yield next.value;
            }
        }
        finally {
            await steps.return();
        }
    }

    // The compression loop. With observe it yields one step per token. Without it never yields,
    // so the only cost over a plain loop is creating the generator.
    async *#compressSteps(uncompressed, windowOverlap, { signal, observe, topK, outputBitsCap }) {
        this.#checkOverlap(windowOverlap);
        const release = this.#claim();
        let finished = false;
        try {
            const model = this.model;
            const tokens = Array.from(model.tokenize(bytesToUtf8(uncompressed), { addBos: false }));
            tokens.push(model.tokenEos());
            let nextTokenIdx = 0;
            const tokenEncoder = new Encoder();
            const builder = observe ? new StepBuilder(model, tokenEncoder, { topK, outputBitsCap }) : null;
            const progress = this.verbose ? new ProgressBar(this.#sinks().err, { total: tokens.length, desc: 'Compressing', unit: 'tok' }) : null;
            let windowIndex = -1;

            while (nextTokenIdx < tokens.length) {
                const startIdx = Math.max(0, nextTokenIdx - windowOverlap);
                let contextLength = this.#startWindow(tokens.slice(startIdx, nextTokenIdx));
                windowIndex++;
                let windowReset = windowIndex > 0;
                for (;;) {
                    const logits = model.logits({ copy: false });
                    if (signal?.aborted && nextTokenIdx < tokens.length - 1) {
                        nextTokenIdx = tokens.length - 1;
                        if (this.verbose) this.#sinks().err.write('\n');
                    }
                    const index = nextTokenIdx++;
                    const nextToken = tokens[index];
                    const cdf = this.computeCdf(logits);
                    tokenEncoder.encodeSymbol(cdf, nextToken);
                    if (builder) {
                        yield builder.build(cdf, nextToken, { index, windowIndex, windowReset, contextLength });
                        windowReset = false;
                    }
                    progress?.update();
                    if (nextTokenIdx === tokens.length || contextLength === model.nCtx) break;
                    model.eval([nextToken]);
                    contextLength++;
                    await this.#maybeYield();
                }
            }
            progress?.close();
            tokenEncoder.finish();
            finished = true;
            return { compressed: tokenEncoder.getEncoded(), totalBits: builder ? builder.cumulativeBits : null, tokens: tokens.length };
        }
        finally {
            if (!finished) this.#clearContext();
            release();
        }
    }

    tokenizerAddsSpacePrefix() {
        const tokenized = this.model.tokenize(new Uint8Array([0x20]), { addBos: false });
        const detokenized = this.model.detokenize(tokenized);
        return detokenized.length === 2 && detokenized[0] === 0x20 && detokenized[1] === 0x20;
    }

    // Throws if `signal` aborts (its reason, or an AbortError), the data ends before an EOS
    // token (the Python tool would loop forever), or more than maxTokens tokens come out.
    // onStep, topK and outputBitsCap are as in compress, and the steps equal the ones compress
    // produced for the same message (the decoder's state is replayed by an encoder that codes
    // the decoded tokens, which is bit-identical to the original one).
    async decompress(compressed, windowOverlap = 0, { signal = null, maxTokens = Infinity, onStep = null, topK, outputBitsCap } = {}) {
        const steps = this.#decompressSteps(compressed, windowOverlap, { signal, maxTokens, observe: onStep !== null, topK, outputBitsCap });
        try {
            for (;;) {
                const next = await steps.next();
                if (next.done) return next.value;
                await onStep(next.value);
            }
        }
        finally {
            await steps.return();
        }
    }

    async *#decompressSteps(compressed, windowOverlap, { signal, maxTokens, observe, topK, outputBitsCap }) {
        this.#checkOverlap(windowOverlap);
        const release = this.#claim();
        let finished = false;
        try {
            const model = this.model;
            const eos = model.tokenEos();
            const seenTokens = [];
            const output = [];
            const tokenDecoder = new Decoder(compressed);
            const shadow = observe ? new Encoder() : null;
            const builder = observe ? new StepBuilder(model, shadow, { topK, outputBitsCap }) : null;
            let windowIndex = -1;
            const utf8Decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
            const bitLimit = compressed.length * 8 + DECODER_LOOKAHEAD_BITS;
            let done = false;
            while (!done) {
                const startIdx = Math.max(0, seenTokens.length - windowOverlap);
                let contextLength = this.#startWindow(seenTokens.slice(startIdx));
                windowIndex++;
                let windowReset = windowIndex > 0;
                for (;;) {
                    const logits = model.logits({ copy: false });
                    if (signal?.aborted) throw signal.reason ?? new DOMException('aborted', 'AbortError');
                    const cdf = this.computeCdf(logits);
                    const nextToken = tokenDecoder.decodeSymbol(cdf);
                    if (nextToken !== eos) {
                        if (tokenDecoder.bitsRead > bitLimit) {
                            throw new Error('compressed data ended before an end-of-stream token (truncated or corrupt data, or a different model, window overlap or format)');
                        }
                        if (seenTokens.length >= maxTokens) throw new Error(`decompression stopped after ${maxTokens} tokens without an end-of-stream token`);
                    }
                    if (builder) {
                        shadow.encodeSymbol(cdf, nextToken);
                        yield builder.build(cdf, nextToken, { index: seenTokens.length, windowIndex, windowReset, contextLength });
                        windowReset = false;
                    }
                    if (nextToken === eos) {
                        done = true;
                        break;
                    }
                    let nextUtf8 = model.detokenize([nextToken]);
                    if (seenTokens.length === 0 && nextUtf8[0] === 0x20 && this.tokenizerAddsSpacePrefix()) {
                        nextUtf8 = nextUtf8.slice(1);
                    }
                    seenTokens.push(nextToken);
                    let text;
                    try {
                        text = utf8Decoder.decode(nextUtf8, { stream: true });
                    }
                    catch {
                        throw new Error('decoded bytes are not valid UTF-8 (corrupt data, or a different model or window overlap)');
                    }
                    const nextBytes = utf8ToBytes(text);
                    for (const byte of nextBytes) output.push(byte);
                    if (this.verbose) this.#sinks().out.write(nextBytes);
                    if (contextLength === model.nCtx) break;
                    model.eval([nextToken]);
                    contextLength++;
                    await this.#maybeYield();
                }
            }
            if (this.verbose) await this.#sinks().out.flush();
            finished = true;
            return Uint8Array.from(output);
        }
        finally {
            if (!finished) this.#clearContext();
            release();
        }
    }
}
