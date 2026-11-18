// Smoke and determinism tests for js/llama.js.
//   tjs run js/test/llama.test.js
// Env: LLAMA_ZIP_TEST_MODEL (GGUF path), LLAMA_LIB (libllama path).
import { LlamaModel } from '../llama.js';
import { test, assert, info, run } from './run.js';

const MODEL_PATH = tjs.env.LLAMA_ZIP_TEST_MODEL
    ?? '/tmp/claude-1000/-home-kaki-Documents-git-github-com-VibedByKaKi-llmzip-desktop/4eff2479-8a67-41fd-9301-2500fcb1ab44/scratchpad/models/SmolLM2-135M.Q8_0.gguf';

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const open = (options = {}) => new LlamaModel({ path: MODEL_PATH, ...options });

const argmax = values => {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
};
const bits = values => new Uint32Array(values.buffer, values.byteOffset, values.length);
const bitIdentical = (a, b) => {
    if (a.length !== b.length) return false;
    const x = bits(a);
    const y = bits(b);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
};
const maxAbsDiff = (a, b) => {
    let max = 0;
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
    return max;
};

const TEXT = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. '.repeat(6)
    + 'The quick brown fox jumps over the lazy dog. Zażółć gęślą jaźń, 日本語のテキスト, 😀.';

// One model for the cheap checks. Tests needing other settings open their own.
const model = open();

test('loads the model', () => {
    assert.ok(model.nVocab > 1000, `nVocab ${model.nVocab}`);
    assert.ok(model.nCtx > 0, `nCtx ${model.nCtx}`);
    info(`nCtx ${model.nCtx}, nVocab ${model.nVocab}, bos ${model.tokenBos()}, eos ${model.tokenEos()}`);
});

test('nCtx can be set explicitly', () => {
    const small = open({ nCtx: 256 });
    assert.equal(small.nCtx, 256);
    small.close();
});

test('missing libllama or model gives a clear error', () => {
    assert.throws(() => open({ path: '/nonexistent/model.gguf' }), /failed to load model/);
});

test('tokenize defaults match llama-cpp-python (no BOS, no special parsing)', () => {
    const plain = model.tokenize('Hello world');
    assert.ok(plain.length >= 2);
    assert.ok(plain instanceof Int32Array);
    const withBos = model.tokenize('Hello world', { addBos: true });
    // SmolLM2 does not configure add_bos, so llama.cpp may add nothing. Either way the text tokens end the list.
    assert.deepEqual(withBos.slice(withBos.length - plain.length), plain);
    const literal = model.tokenize('<|endoftext|>');
    const parsed = model.tokenize('<|endoftext|>', { special: true });
    assert.ok(literal.length > 1, 'special=false must split the marker into text tokens');
    assert.deepEqual(parsed, [model.tokenEos()]);
    assert.equal(model.tokenize('').length, 0);
});

test('eval then logits has nVocab entries and a plausible greedy token', () => {
    model.reset();
    model.eval([model.tokenBos(), ...model.tokenize('The capital of France is')]);
    const logits = model.logits();
    assert.equal(logits.length, model.nVocab);
    assert.ok(logits.every(Number.isFinite), 'non-finite logit');
    const next = decoder.decode(model.detokenize([argmax(logits)]));
    info(`greedy next token: ${JSON.stringify(next)}`);
    assert.ok(/Paris/.test(next), `expected Paris, got ${JSON.stringify(next)}`);
});

test('nPast tracks eval and reset, and eval refuses to overflow the context', () => {
    model.reset();
    assert.equal(model.nPast, 0);
    model.eval(model.tokenize('Hello world'));
    assert.equal(model.nPast, model.tokenize('Hello world').length);
    model.reset();
    assert.equal(model.nPast, 0);
    const small = open({ nCtx: 300 });
    // llama.cpp pads the context up to a multiple of 256 but clamps n_batch to the requested size.
    assert.equal(small.nCtx, 512);
    assert.equal(small.nBatch, 300);
    assert.throws(() => small.eval(new Int32Array(513)), /exceed the context/);
    small.eval(new Int32Array(512));
    assert.equal(small.nPast, 512);
    small.close();
});

test('logits() copies by default and a view aliases the live buffer', () => {
    model.reset();
    model.eval(model.tokenize('Hello'));
    const copy = model.logits();
    const view = model.logits({ copy: false });
    assert.ok(bitIdentical(copy, view));
    model.eval(model.tokenize(' world'));
    assert.ok(!bitIdentical(copy, model.logits()), 'copy must not change after the next eval');
});

test('detokenize(tokenize(x)) round trips', () => {
    for (const text of ['Hello world', ' leading space', 'trailing space ', 'line\nbreak\ttab', 'ünïcödé café', '日本語のテキスト', '😀 emoji', 'Zażółć gęślą jaźń', '']) {
        const tokens = model.tokenize(encoder.encode(text));
        assert.equal(decoder.decode(model.detokenize(tokens)), text, `round trip of ${JSON.stringify(text)}`);
    }
});

test('detokenize keeps raw bytes of split multibyte characters', () => {
    // The emoji is a few byte-level tokens. Each alone may be invalid UTF-8, all together restore it.
    const tokens = model.tokenize('😀');
    const joined = model.detokenize(tokens);
    assert.deepEqual(joined, encoder.encode('😀'));
    let total = 0;
    for (const token of tokens) total += model.detokenize([token]).length;
    assert.equal(total, joined.length);
});

test('tokenizer space prefix probe (llama_zip.py tokenizer_adds_space_prefix)', () => {
    const tokens = model.tokenize(' ');
    const bytes = model.detokenize(tokens);
    info(`tokenize(" ") -> ${Array.from(tokens)}, detokenize -> ${JSON.stringify(decoder.decode(bytes))}, adds space prefix: ${decoder.decode(bytes) === '  '}`);
    assert.equal(decoder.decode(bytes), ' ');
});

test('detokenize strips one leading space only after BOS (llama-cpp-python rule)', () => {
    const [first] = model.tokenize(' Hello');
    const text = decoder.decode(model.detokenize([first]));
    assert.equal(decoder.decode(model.detokenize([model.tokenBos(), first])), text.startsWith(' ') ? text.slice(1) : text);
});

test('reset gives bit-identical logits for the same tokens', () => {
    const tokens = [model.tokenBos(), ...model.tokenize(TEXT)];
    model.reset();
    model.eval(tokens);
    const first = model.logits();
    // Disturb the context in between.
    model.reset();
    model.eval(model.tokenize('something unrelated entirely'));
    model.reset();
    model.eval(tokens);
    const second = model.logits();
    assert.ok(bitIdentical(first, second), `max abs diff ${maxAbsDiff(first, second)}`);
    info(`${tokens.length} tokens, logits bit-identical after reset`);
});

test('token by token replay (encoder vs decoder pattern) is bit-identical at every step', () => {
    const tokens = [model.tokenBos(), ...model.tokenize(TEXT)];
    const record = () => {
        model.reset();
        const all = [];
        for (const token of tokens) {
            model.eval([token]);
            all.push(model.logits());
        }
        return all;
    };
    const a = record();
    const b = record();
    for (let i = 0; i < a.length; i++) assert.ok(bitIdentical(a[i], b[i]), `step ${i} differs, max abs diff ${maxAbsDiff(a[i], b[i])}`);
    info(`${a.length} steps identical`);
});

test('a second model instance reproduces the logits bit for bit', () => {
    const tokens = [model.tokenBos(), ...model.tokenize(TEXT)];
    model.reset();
    model.eval(tokens);
    const reference = model.logits();
    const other = open();
    other.eval(tokens);
    assert.ok(bitIdentical(reference, other.logits()), `max abs diff ${maxAbsDiff(reference, other.logits())}`);
    other.close();
});

// Findings, not requirements: different batch shapes and thread counts use different kernels
// or reduction orders, so differences are expected in principle. The output tells which ones occur.
test('report: logits across n_batch chunkings and thread counts', () => {
    const full = [model.tokenBos(), ...model.tokenize(TEXT)];
    for (const length of [20, full.length]) {
        const tokens = full.slice(0, length);
        const logitsFor = options => {
            const variant = open(options);
            variant.eval(tokens);
            const logits = variant.logits();
            variant.close();
            return logits;
        };
        const reference = logitsFor({ nBatch: 512, nThreads: 4 });
        info(`prefix of ${tokens.length} tokens, reference nBatch 512 (one chunk), 4 threads`);
        for (const nBatch of [256, 64, 7, 1]) {
            const logits = logitsFor({ nBatch, nThreads: 4 });
            info(`  nBatch ${nBatch}: bit-identical ${bitIdentical(reference, logits)}, max abs diff ${maxAbsDiff(reference, logits)}, same argmax ${argmax(reference) === argmax(logits)}`);
        }
        for (const nThreads of [1, 2, 8]) {
            const logits = logitsFor({ nBatch: 512, nThreads });
            info(`  nThreads ${nThreads}: bit-identical ${bitIdentical(reference, logits)}, max abs diff ${maxAbsDiff(reference, logits)}`);
        }
        // Several eval() calls on one model, which is what the coder does when it replays a window.
        const variant = open({ nThreads: 4 });
        variant.eval(tokens.slice(0, 10));
        variant.eval(tokens.slice(10));
        info(`  eval split 10 + rest: bit-identical ${bitIdentical(reference, variant.logits())}, max abs diff ${maxAbsDiff(reference, variant.logits())}`);
        variant.close();
    }
});

test('close is idempotent and later calls throw', () => {
    const temporary = open();
    temporary.close();
    temporary.close();
    assert.throws(() => temporary.tokenize('x'), /closed/);
});

test('closing the shared model last leaves nothing to crash at exit', () => {
    model.close();
});

await run();
