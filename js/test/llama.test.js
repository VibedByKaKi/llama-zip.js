// Smoke and determinism tests for js/llama.js.
//   tjs run js/test/llama.test.js
// Env: LLAMA_ZIP_TEST_MODEL (GGUF path), LLAMA_LIB (libllama path).
import { LlamaModel, ModelParams, ContextParams, Batch, KNOWN_VERSIONS, checkVersion, libraryInfo } from '../llama.js';
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

// Expected layouts of llama.h 1.0.0 on LP64 (x86-64 and arm64 Linux and macOS). Generated from the
// real header, not from this binding: a C program included /home/linuxbrew/.linuxbrew/include/llama.h
// and printed sizeof and offsetof for every field (gcc -I<include dir> probe.c && ./probe).
// Regenerate it the same way for a new llama.cpp version before adding it to KNOWN_VERSIONS.
// Windows (LLP64) has the same layout, because no field is a long.
const LAYOUTS = {
    llama_model_params: { size: 80, fields: {
        devices: 0, tensor_buft_overrides: 8, n_gpu_layers: 16, split_mode: 20, load_mode: 24, lazy_mode: 28,
        main_gpu: 32, tensor_split: 40, progress_callback: 48, progress_callback_user_data: 56, kv_overrides: 64,
        vocab_only: 72, check_tensors: 73, use_extra_bufts: 74, no_host: 75, no_alloc: 76, load_mtp: 77
    } },
    llama_context_params: { size: 176, fields: {
        n_ctx: 0, n_batch: 4, n_ubatch: 8, n_seq_max: 12, n_rs_seq: 16, n_outputs_max: 20, n_outputs_max_per_seq: 24,
        n_threads: 28, n_threads_batch: 32, ctx_type: 36, rope_scaling_type: 40, pooling_type: 44, attention_type: 48,
        flash_attn_type: 52, rope_freq_base: 56, rope_freq_scale: 60, yarn_ext_factor: 64, yarn_attn_factor: 68,
        yarn_beta_fast: 72, yarn_beta_slow: 76, yarn_orig_ctx: 80, cb_eval: 88, cb_eval_user_data: 96, type_k: 104,
        type_v: 108, moe_cache_size: 112, moe_cache_cpu_share: 120, abort_callback: 128, abort_callback_data: 136,
        embeddings: 144, offload_kqv: 145, no_perf: 146, op_offload: 147, swa_full: 148, kv_unified: 149,
        samplers: 152, n_samplers: 160, ctx_other: 168
    } },
    llama_batch: { size: 56, fields: { n_tokens: 0, token: 8, embd: 16, pos: 24, n_seq_id: 32, seq_id: 40, logits: 48 } }
};

for (const [name, struct] of [['llama_model_params', ModelParams], ['llama_context_params', ContextParams], ['llama_batch', Batch]])
    test(`${name} layout equals the C compiler's (size and every offset)`, () => {
        const expected = LAYOUTS[name];
        const actual = Object.fromEntries(struct.describe().map(field => [field.name, field.offset]));
        assert.equal(struct.size, expected.size, 'sizeof');
        assert.deepEqual(actual, expected.fields);
    });

test('library version check: known, unknown, missing llama_version, override', () => {
    assert.deepEqual(Object.keys(KNOWN_VERSIONS), ['1.0.0']);
    assert.equal(checkVersion(() => '1.0.0', 'lib').build, 12359);
    assert.throws(() => checkVersion(undefined, '/old/libllama.so'), /needs llama\.cpp 1\.0\.0 \(build 12359\) or newer layout-compatible/);
    assert.throws(() => checkVersion(() => '1.1.0', 'lib'), /1\.1\.0.*LLAMA_ZIP_ALLOW_UNTESTED=1/);
    tjs.env.LLAMA_ZIP_ALLOW_UNTESTED = '1';
    try {
        assert.equal(checkVersion(() => '1.1.0', 'lib').version, '1.1.0');
        assert.throws(() => checkVersion(undefined, 'lib'), /needs llama\.cpp/);
    }
    finally {
        delete tjs.env.LLAMA_ZIP_ALLOW_UNTESTED;
    }
});

test('the loaded library reports a known version and the model exposes it', () => {
    assert.ok(Object.hasOwn(KNOWN_VERSIONS, model.llamaVersion) || tjs.env.LLAMA_ZIP_ALLOW_UNTESTED === '1', model.llamaVersion);
    assert.ok(model.llamaBuild.startsWith(model.llamaVersion));
    assert.equal(libraryInfo().text, model.llamaBuild);
    info(`${model.llamaBuild}; ${model.systemInfo.trim()}`);
});

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
