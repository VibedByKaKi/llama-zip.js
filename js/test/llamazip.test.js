// Port of tests/python/test_llamazip.py. Run with `tjs run js/test/llamazip.test.js`.
//
// The Python SIGINT tests raise a real signal from the fake model. In this port the library
// takes an AbortSignal and the CLI maps SIGINT to it (see cli.test.js), so the same scenarios
// abort an AbortController from the same hook points.
import { Encoder } from '../arithmetic.js';
import { DEFAULT_N_BATCH, DEFAULT_N_THREADS, LlamaZip, PUA_START } from '../llamazip.js';
import { BOS, EOS, FakeModel, VOCAB_SIZE, captureIo, createModel, fakeTest as test } from './fake_llama.js';
import { assert, run } from './run.js';
import { bytesOf, fromHex, loadVectors, toHex } from './vectors.js';

const VECTORS = await loadVectors('llamazip_fake_model.json');

const caseId = c => `${c.space_prefix ? 'space' : 'nospace'}-ctx${c.n_ctx}-ov${c.window_overlap}-${c.name}`;

const zipper = (options = {}) => new LlamaZip('fake.gguf', { nCtx: 32, createModel, ...options });

const make = (nCtx, spacePrefix = false, options = {}) => {
    FakeModel.spacePrefix = spacePrefix;
    return new LlamaZip('fake.gguf', { nCtx, createModel, ...options });
};

const text = value => bytesOf(value);
const same = (actual, expected, message) => assert.equal(toHex(actual), toHex(expected), message);

for (const c of VECTORS.cases) {
    test(`compress matches vector [${caseId(c)}]`, async () => {
        const z = make(c.n_ctx, c.space_prefix);
        assert.equal(toHex(await z.compress(fromHex(c.input_hex), c.window_overlap)), c.compressed_hex);
    });
}

for (const c of VECTORS.cases) {
    test(`decompress matches vector [${caseId(c)}]`, async () => {
        const z = make(c.n_ctx, c.space_prefix);
        same(await z.decompress(fromHex(c.compressed_hex), c.window_overlap), fromHex(c.input_hex));
    });
}

test('fake model description matches the fake', () => {
    assert.equal(VECTORS.model.vocab_size, VOCAB_SIZE);
    assert.deepEqual([VECTORS.model.bos, VECTORS.model.eos], [BOS, EOS]);
});

// TestLoadModel

test('LoadModel passes arguments to the model', () => {
    new LlamaZip('some/model.gguf', { nCtx: 100, nGpuLayers: 3, useMlock: true, createModel });
    assert.equal(FakeModel.instances.length, 1);
    const [model] = FakeModel.instances;
    assert.equal(model.path, 'some/model.gguf');
    assert.equal(model.nCtx, 100);
    assert.equal(model.nGpuLayers, 3);
    assert.equal(model.useMlock, true);
});

test('LoadModel defaults', () => {
    new LlamaZip('m.gguf', { createModel });
    const [model] = FakeModel.instances;
    assert.equal(model.nGpuLayers, -1);
    assert.equal(model.useMlock, false);
    assert.equal(model.nCtx, 64);
});

test('LoadModel passes the thread count and batch size, with fixed defaults', () => {
    new LlamaZip('m.gguf', { createModel });
    new LlamaZip('m.gguf', { createModel, nThreads: 2, nBatch: 16 });
    const [first, second] = FakeModel.instances;
    assert.equal(first.nThreads, DEFAULT_N_THREADS);
    assert.equal(first.nBatch, DEFAULT_N_BATCH);
    assert.equal(DEFAULT_N_THREADS, 4);
    assert.equal(second.nThreads, 2);
    assert.equal(second.nBatch, 16);
});

test('LoadModel is quiet by default', () => {
    const io = captureIo();
    new LlamaZip('m.gguf', { createModel, io });
    assert.equal(io.errText(), '');
    assert.equal(io.outBytes().length, 0);
});

test('LoadModel verbose shows and clears the loading message', () => {
    const io = captureIo();
    const z = new LlamaZip('m.gguf', { createModel, io, verbose: true });
    assert.equal(z.verbose, true);
    assert.equal(io.errText(), 'Loading model...' + '\r' + ' '.repeat('Loading model...'.length) + '\r');
});

test('LoadModel.load shows and clears the loading message too', async () => {
    const io = captureIo();
    const z = await LlamaZip.load('m.gguf', { createModel, io, verbose: true });
    assert.equal(io.errText(), 'Loading model...' + '\r' + ' '.repeat('Loading model...'.length) + '\r');
    assert.ok(z instanceof LlamaZip);
});

// TestRoundTrip

const ROUND_TRIP_DATA = [
    new Uint8Array(0),
    text('a'),
    text('the thing'),
    text('  leading spaces'),
    text('café € \u{1f600}'),
    Uint8Array.from({ length: 256 }, (_, i) => i),
    Uint8Array.of(0xff, 0xfe, 0x80),
    text('the '.repeat(100)),
];

for (const [index, data] of ROUND_TRIP_DATA.entries()) {
    for (const overlap of [0, 1, 7, 31]) {
        test(`RoundTrip round trip [data ${index} (${data.length} bytes), overlap ${overlap}]`, async () => {
            const z = zipper();
            const compressed = await z.compress(data, overlap);
            same(await z.decompress(compressed, overlap), data);
        });
    }
}

test('RoundTrip returns Uint8Array', async () => {
    const z = zipper();
    assert.ok((await z.compress(text('abc'))) instanceof Uint8Array);
    assert.ok((await z.decompress(await z.compress(text('abc')))) instanceof Uint8Array);
});

test('RoundTrip default window overlap is zero', async () => {
    const z = zipper();
    const data = text('the thing '.repeat(10));
    same(await z.compress(data), await z.compress(data, 0));
    same(await z.decompress(await z.compress(data)), await z.decompress(await z.compress(data, 0), 0));
});

test('RoundTrip compression is deterministic', async () => {
    const z = zipper();
    const data = text('the king the thing');
    same(await z.compress(data), await z.compress(data));
});

test('RoundTrip empty input is just the EOS symbol', async () => {
    const z = zipper();
    const compressed = await z.compress(new Uint8Array(0));
    assert.equal((await z.decompress(compressed)).length, 0);
    assert.ok(compressed.length <= 5);
});

test('RoundTrip model is reset each time', async () => {
    const z = zipper();
    const model = FakeModel.instances[0];
    await z.compress(text('ab'));
    await z.decompress(await z.compress(text('ab')));
    assert.equal(model.resetCalls, 3);
});

test('RoundTrip overlap changes output once the window fills', async () => {
    const z = zipper();
    const data = text('the thing, the king; '.repeat(6));
    const a = await z.compress(data, 0);
    const b = await z.compress(data, 10);
    assert.notEqual(toHex(a), toHex(b));
    same(await z.decompress(a, 0), data);
    same(await z.decompress(b, 10), data);
});

test('RoundTrip overlap is irrelevant while the window has room', async () => {
    const z = zipper();
    same(await z.compress(text('ab'), 0), await z.compress(text('ab'), 20));
});

// TestCompress

const spyOnSymbols = () => {
    const symbols = [];
    const original = Encoder.prototype.encodeSymbol;
    Encoder.prototype.encodeSymbol = function (cdf, symbol) {
        symbols.push(symbol);
        return original.call(this, cdf, symbol);
    };
    return { symbols, restore: () => { Encoder.prototype.encodeSymbol = original; } };
};

const withSpy = async fn => {
    const spy = spyOnSymbols();
    try {
        await fn(spy.symbols);
    }
    finally {
        spy.restore();
    }
};

test('Compress encodes token ids followed by EOS', () => withSpy(async symbols => {
    const z = zipper();
    await z.compress(text('the x'));
    assert.deepEqual(symbols, [...FakeModel.instances[0].tokenize(text('the x')), EOS]);
}));

test('Compress empty input encodes only EOS', () => withSpy(async symbols => {
    await zipper().compress(new Uint8Array(0));
    assert.deepEqual(symbols, [EOS]);
}));

test('Compress bytes are mapped to UTF-8 before tokenizing', () => withSpy(async symbols => {
    const z = zipper();
    await z.compress(Uint8Array.of(0xff));
    const pua = text(String.fromCodePoint(PUA_START + 0xFF));
    const expected = [...FakeModel.instances[0].tokenize(pua), EOS];
    assert.deepEqual(symbols, expected);
    assert.equal(expected.length, 4); // three single byte tokens plus EOS
}));

test('Compress context starts with BOS and grows', async () => {
    await zipper().compress(text('abc'));
    assert.deepEqual(FakeModel.instances[0].windowPrefixes, [[BOS]]);
});

test('Compress window restarts with overlap tokens', async () => {
    const z = make(8);
    const data = text('abcdefghijklmnopqrstuvwxyz');
    const tokens = Array.from(z.model.tokenize(data));
    await z.compress(data, 3);
    const calls = z.model.windowPrefixes;
    assert.deepEqual(calls[0], [BOS]);
    assert.ok(calls.length > 3);
    let consumed = 0;
    for (const call of calls) {
        // A window of 8 holds BOS, up to 3 overlap tokens and the new ones.
        const start = Math.max(0, consumed - 3);
        assert.deepEqual(call, [BOS, ...tokens.slice(start, consumed)]);
        // Sampling continues until the context holds n_ctx tokens, inclusive.
        consumed += 8 - call.length + 1;
    }
});

test('Compress window with zero overlap starts from BOS only', async () => {
    const z = make(6);
    await z.compress(text('abcdefghijklmn'), 0);
    assert.ok(z.model.windowPrefixes.every(call => call.length === 1 && call[0] === BOS));
    assert.ok(z.model.windowPrefixes.length >= 3);
});

test('Compress context never exceeds n_ctx', async () => {
    const z = make(8);
    const seen = [];
    FakeModel.onLogits = context => seen.push(context.length);
    await z.compress(text('abcdefghijklmnopqrstuvwxyz'), 5);
    assert.equal(Math.max(...seen), 8);
});

test('Compress maximum overlap still makes progress', async () => {
    const z = make(8);
    const data = text('abcdefghijkl');
    same(await z.decompress(await z.compress(data, 7), 7), data);
});

test('Compress is silent without verbose', async () => {
    const io = captureIo();
    await new LlamaZip('m.gguf', { nCtx: 32, createModel, io }).compress(text('the thing'));
    assert.equal(io.errText(), '');
    assert.equal(io.outBytes().length, 0);
});

test('Compress verbose uses a progress bar', async () => {
    const io = captureIo();
    await new LlamaZip('m.gguf', { nCtx: 32, createModel, io, verbose: true }).compress(text('the thing'));
    assert.ok(io.errText().includes('Compressing'));
});

test('Compress rejects a window overlap outside [0, nCtx - 1]', async () => {
    const z = make(8);
    for (const overlap of [-1, 8, 1.5, NaN]) {
        await assert.rejects(() => z.compress(text('abc'), overlap), /windowOverlap must be an integer in the range \[0, 7\]/);
        await assert.rejects(() => z.decompress(text('abc'), overlap), /windowOverlap must be an integer in the range \[0, 7\]/);
    }
});

// TestDecompress

test('Decompress stops at EOS and excludes it', async () => {
    const z = zipper();
    same(await z.decompress(await z.compress(text('ab'))), text('ab'));
});

test('Decompress EOS only stream', async () => {
    const z = zipper();
    assert.equal((await z.decompress(await z.compress(new Uint8Array(0)))).length, 0);
});

test('Decompress trailing garbage after the stream is ignored', async () => {
    const z = zipper();
    const compressed = await z.compress(text('the king'));
    same(await z.decompress(Uint8Array.from([...compressed, ...new Uint8Array(8)])), text('the king'));
});

test('Decompress verbose streams output to stdout', async () => {
    const io = captureIo();
    const z = new LlamaZip('m.gguf', { nCtx: 32, createModel, io, verbose: true });
    const compressed = await z.compress(text('the thing'));
    io.clear();
    const result = await z.decompress(compressed);
    same(io.outBytes(), text('the thing'));
    same(result, text('the thing'));
});

test('Decompress non verbose does not stream', async () => {
    const io = captureIo();
    const z = new LlamaZip('m.gguf', { nCtx: 32, createModel, io });
    await z.decompress(await z.compress(text('the thing')));
    assert.equal(io.outBytes().length, 0);
});

test('Decompress multibyte character split across tokens', async () => {
    const z = zipper();
    const data = text('€\u{1f600}'); // single byte tokens split each character
    same(await z.decompress(await z.compress(data)), data);
});

test('Decompress invalid bytes survive', async () => {
    const z = zipper();
    const data = Uint8Array.of(0x61, 0xc3, 0xff, 0x62, 0x80);
    same(await z.decompress(await z.compress(data)), data);
});

test('Decompress PUA collisions survive', async () => {
    const z = zipper();
    const data = text('ab');
    same(await z.decompress(await z.compress(data)), data);
});

test('Decompress window restarts use overlap tokens', async () => {
    const z = make(8);
    const data = text('abcdefghijklmnopqrstuvwxyz');
    const compressed = await z.compress(data, 3);
    z.model.windowPrefixes.length = 0;
    same(await z.decompress(compressed, 3), data);
    const calls = z.model.windowPrefixes;
    assert.deepEqual(calls[0], [BOS]);
    assert.ok(calls.every(call => call[0] === BOS));
    assert.equal(Math.max(...calls.slice(1).map(call => call.length)), 1 + 3);
});

test('Decompress mismatched overlap gives a different stream', async () => {
    const z = zipper();
    const data = text('the thing, the king; '.repeat(6));
    assert.notEqual(toHex(await z.compress(data, 4)), toHex(await z.compress(data, 0)));
});

// Additions: the Python decoder loops forever on data without an EOS, the port stops.

test('Decompress throws on data that ends before an EOS token', async () => {
    const z = zipper();
    for (const data of [new Uint8Array(0), Uint8Array.of(0), Uint8Array.of(0x12, 0x34)]) {
        await assert.rejects(() => z.decompress(data), /ended before an end-of-stream token/);
    }
});

test('Decompress throws on a stream cut short', async () => {
    const z = zipper();
    const compressed = await z.compress(text('the thing, the king; '.repeat(4)));
    await assert.rejects(() => z.decompress(compressed.slice(0, 3)), /ended before an end-of-stream token|not valid UTF-8/);
});

test('Decompress never trips the data exhaustion check on valid streams', async () => {
    const z = make(16);
    for (let size = 0; size < 60; size++) {
        const data = Uint8Array.from({ length: size }, (_, i) => (i * 7 + size) % 256);
        for (const overlap of [0, 5, 15]) same(await z.decompress(await z.compress(data, overlap), overlap), data);
    }
});

test('Decompress maxTokens stops a long stream with a clear error', async () => {
    const z = zipper();
    const compressed = await z.compress(text('abcdefghij'));
    await assert.rejects(() => z.decompress(compressed, 0, { maxTokens: 3 }), /stopped after 3 tokens/);
    same(await z.decompress(compressed, 0, { maxTokens: 10 }), text('abcdefghij'));
});

test('Decompress rethrows the abort reason of its signal', async () => {
    const z = zipper();
    const compressed = await z.compress(text('abcdefghij'));
    const controller = new AbortController();
    const reason = new Error('stop now');
    FakeModel.onLogits = context => {
        if (context.length === 3) controller.abort(reason);
    };
    await assert.rejects(() => z.decompress(compressed, 0, { signal: controller.signal }), /stop now/);
});

// TestSpacePrefix

test('SpacePrefix detects a model that adds a prefix', () => {
    assert.ok(make(32, true).tokenizerAddsSpacePrefix() === true);
});

test('SpacePrefix detects a model without a prefix', () => {
    assert.ok(make(32, false).tokenizerAddsSpacePrefix() === false);
});

test('SpacePrefix first token space is stripped when the tokenizer adds one', async () => {
    const z = make(32, true);
    same(z.model.detokenize(z.model.tokenize(text('x'))), text(' x'));
    same(await z.decompress(await z.compress(text('x'))), text('x'));
});

test('SpacePrefix leading space in the input is kept', async () => {
    const z = make(32, true);
    for (const data of [text(' x'), text('  x'), text(' the thing')]) same(await z.decompress(await z.compress(data)), data);
});

test('SpacePrefix only the first token is stripped', async () => {
    const z = make(32, true);
    const data = text('the  thing   x');
    same(await z.decompress(await z.compress(data)), data);
});

test('SpacePrefix without the special case the prefix would leak', async () => {
    const compressed = await make(32, true).compress(text('x'));
    // A model that does not add the prefix keeps the space it decoded.
    same(await make(32, false).decompress(compressed), text(' x'));
});

test('SpacePrefix stripping applies to the first token of each decompress', async () => {
    const z = make(32, true);
    const compressed = await z.compress(text('the thing'));
    same(await z.decompress(compressed), text('the thing'));
    same(await z.decompress(compressed), text('the thing'));
});

test('SpacePrefix window restart does not strip again', async () => {
    const z = make(8, true);
    const data = text('a b c d e f g h i j k l m n o p');
    same(await z.decompress(await z.compress(data, 2), 2), data);
});

test('SpacePrefix probe runs once and only for a first token with a space', async () => {
    const calls = [];
    const original = LlamaZip.prototype.tokenizerAddsSpacePrefix;
    LlamaZip.prototype.tokenizerAddsSpacePrefix = function () {
        calls.push(1);
        return original.call(this);
    };
    try {
        const plain = make(32, false);
        await plain.decompress(await plain.compress(text('abc')));
        assert.deepEqual(calls, []); // the first token "a" has no space to strip
        const prefixed = make(32, true);
        await prefixed.decompress(await prefixed.compress(text('abc')));
        assert.deepEqual(calls, [1]); // only the first token is probed
    }
    finally {
        LlamaZip.prototype.tokenizerAddsSpacePrefix = original;
    }
});

// TestSigint, with an AbortSignal in place of the process signal

const countingAbort = (controller, target) => {
    let count = 0;
    FakeModel.onLogits = () => {
        count++;
        if (count === target) controller.abort();
    };
};

test('Sigint interrupt truncates the stream but stays decodable', async () => {
    const z = make(64);
    const data = text('abcdefghijklmnop');
    const controller = new AbortController();
    countingAbort(controller, 4);
    const compressed = await z.compress(data, 0, { signal: controller.signal });
    FakeModel.onLogits = null;
    same(await z.decompress(compressed), data.slice(0, 3));
});

test('Sigint without a signal nothing is interrupted and no global handler is needed', async () => {
    const z = make(64);
    const data = text('abc');
    same(await z.compress(data, 0, { signal: new AbortController().signal }), await z.compress(data));
});

test('Sigint abort at every step is handled and decodes to nothing', async () => {
    const z = make(64);
    const controller = new AbortController();
    FakeModel.onLogits = () => controller.abort();
    const compressed = await z.compress(text('abcdef'), 0, { signal: controller.signal });
    FakeModel.onLogits = null;
    assert.equal((await z.decompress(compressed)).length, 0);
});

test('Sigint interrupt on the last token changes nothing', async () => {
    const z = make(64);
    const data = text('abc');
    const expected = await z.compress(data);
    const controller = new AbortController();
    countingAbort(controller, 4); // the EOS step
    same(await z.compress(data, 0, { signal: controller.signal }), expected);
});

test('Sigint verbose interrupt prints a newline', async () => {
    const io = captureIo();
    const z = new LlamaZip('m.gguf', { nCtx: 64, createModel, io, verbose: true });
    const controller = new AbortController();
    countingAbort(controller, 2);
    await z.compress(text('abcdef'), 0, { signal: controller.signal });
    assert.ok(io.errText().includes('\n'));
});

test('Sigint interrupt across windows', async () => {
    const z = make(6);
    const controller = new AbortController();
    countingAbort(controller, 9);
    const compressed = await z.compress(text('abcdefghijklmnopqrst'), 2, { signal: controller.signal });
    FakeModel.onLogits = null;
    same(await z.decompress(compressed, 2), text('abcdefgh'));
});

test('vocab logits helper shapes', () => {
    assert.equal(VOCAB_SIZE, VECTORS.model.vocab_size);
    const z = zipper();
    assert.equal(z.model.nVocab, VOCAB_SIZE);
    const cdf = z.computeCdf(Float32Array.of(0, 0));
    assert.deepEqual(cdf, [2 ** 31, 2 ** 32]);
});

await run();
