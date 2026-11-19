// Deterministic stand-in for LlamaModel (js/llama.js), the port of tests/python/fake_llama.py.
//
// The model is a byte-level tokenizer with a few merges and a logits function that
// depends only on the context. Both are plain integer arithmetic, as described in
// tests/vectors/llamazip_fake_model.json.

import { test } from './run.js';

export const BOS = 0;
export const EOS = 1;
export const BYTE_BASE = 2;
export const MERGES = ['20746865', '746865', '696e67', '6865', '7468', '2020', '20'].map(hex => {
    const bytes = [];
    for (let i = 0; i < hex.length; i += 2) bytes.push(parseInt(hex.slice(i, i + 2), 16));
    return Uint8Array.from(bytes);
});
export const DEFAULT_N_CTX = 64;

export const VOCAB = [new Uint8Array(0), new Uint8Array(0), ...Array.from({ length: 256 }, (_, i) => Uint8Array.of(i)), ...MERGES];
export const VOCAB_SIZE = VOCAB.length;

export const fakeLogits = context => {
    const last = context.length ? context[context.length - 1] : 0;
    const before = context.length > 1 ? context[context.length - 2] : 0;
    const h = last * 131 + before * 31 + context.length * 7;
    return Float32Array.from({ length: VOCAB_SIZE }, (_, i) => ((h * (i + 3) + i * i) % 31) / 2);
};

const startsWithAt = (text, prefix, index) => {
    for (let k = 0; k < prefix.length; k++) if (text[index + k] !== prefix[k]) return false;
    return true;
};

const byLength = MERGES.map((_, k) => k).sort((a, b) => MERGES[b].length - MERGES[a].length);

export class FakeModel {
    static instances = [];
    static spacePrefix = false;
    static onLogits = null;

    static reset() {
        FakeModel.instances = [];
        FakeModel.spacePrefix = false;
        FakeModel.onLogits = null;
    }

    // Takes the LlamaModel constructor options.
    constructor({ path, nCtx = 0, nGpuLayers = 0, useMlock = false, nThreads = 0, nBatch = 512 } = {}) {
        this.path = path;
        this.useMlock = useMlock;
        this.nGpuLayers = nGpuLayers;
        this.nThreads = nThreads;
        this.nBatch = nBatch;
        this.nCtx = nCtx || DEFAULT_N_CTX;
        this.nVocab = VOCAB_SIZE;
        this.resetCalls = 0;
        // The tokens of the first eval() after each reset(), which is where a window starts.
        this.windowPrefixes = [];
        this.context = [];
        this.closed = false;
        this.awaitingPrefix = true;
        FakeModel.instances.push(this);
    }

    get nPast() {
        return this.context.length;
    }

    tokenBos() {
        return BOS;
    }

    tokenEos() {
        return EOS;
    }

    reset() {
        this.resetCalls++;
        this.context = [];
        this.awaitingPrefix = true;
    }

    tokenize(data, { addBos = false } = {}) {
        let text = typeof data === 'string' ? new TextEncoder().encode(data) : data;
        if (FakeModel.spacePrefix && text.length) text = Uint8Array.from([0x20, ...text]);
        const tokens = addBos ? [BOS] : [];
        let i = 0;
        while (i < text.length) {
            const k = byLength.find(candidate => startsWithAt(text, MERGES[candidate], i));
            if (k !== undefined) {
                tokens.push(BYTE_BASE + 256 + k);
                i += MERGES[k].length;
            }
            else {
                tokens.push(BYTE_BASE + text[i]);
                i++;
            }
        }
        return Int32Array.from(tokens);
    }

    detokenize(tokens) {
        const pieces = Array.from(tokens, token => VOCAB[token]);
        const output = new Uint8Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
        let offset = 0;
        for (const piece of pieces) {
            output.set(piece, offset);
            offset += piece.length;
        }
        return output;
    }

    eval(tokens) {
        if (this.context.length + tokens.length > this.nCtx) throw new Error('Requested tokens exceed context window');
        if (this.awaitingPrefix) {
            this.windowPrefixes.push(Array.from(tokens));
            this.awaitingPrefix = false;
        }
        this.context.push(...tokens);
    }

    logits() {
        FakeModel.onLogits?.([...this.context]);
        return fakeLogits(this.context);
    }

    close() {
        this.closed = true;
    }
}

export const createModel = options => new FakeModel(options);


// Registers a test that starts and ends with a clean FakeModel (the autouse fixture in conftest.py).
export const fakeTest = (name, fn) => test(name, async () => {
    FakeModel.reset();
    try {
        await fn();
    }
    finally {
        FakeModel.reset();
    }
});

// Sinks that collect what a LlamaZip or main() writes, like pytest's capfd and capsysbinary.
export const captureIo = () => {
    const chunks = { out: [], err: [] };
    const sink = list => ({
        write: data => chunks[list].push(typeof data === 'string' ? new TextEncoder().encode(data) : Uint8Array.from(data)),
        flush: async () => {},
    });
    const join = list => {
        const parts = chunks[list];
        const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
        let offset = 0;
        for (const part of parts) {
            bytes.set(part, offset);
            offset += part.length;
        }
        return bytes;
    };
    return {
        out: sink('out'),
        err: sink('err'),
        outBytes: () => join('out'),
        errText: () => new TextDecoder().decode(join('err')),
        clear: () => {
            chunks.out.length = 0;
            chunks.err.length = 0;
        },
    };
};
