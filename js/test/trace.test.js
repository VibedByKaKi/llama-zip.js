// Tests for the step API (js/trace.js, onStep, trace() and --trace). Run with `tjs run js/test/trace.test.js`.
//
// Most tests use the fake model. The last one uses the real model and is skipped without libllama
// or the model file (LLAMA_ZIP_TEST_MODEL).
import { main } from '../cli.js';
import { LlamaZip, computeCdf } from '../llamazip.js';
import { selectTopTokens } from '../trace.js';
import { FakeModel, captureIo, createModel, fakeTest as test } from './fake_llama.js';
import { assert, info, run, skip, test as plainTest } from './run.js';
import { bytesOf, toHex } from './vectors.js';

const TEXT = bytesOf('the thing, in the hearing of the other things. '.repeat(4));
const SHORT = bytesOf('the thing');

const makeZip = (options = {}) => new LlamaZip('fake.gguf', { nCtx: 64, createModel, ...options });
const collect = async (zip, data, overlap = 0, options = {}) => {
    const steps = [];
    const compressed = await zip.compress(data, overlap, { ...options, onStep: step => void steps.push(step) });
    return { steps, compressed };
};
const near = (actual, expected, tolerance, message = '') => assert.ok(Math.abs(actual - expected) <= tolerance, `${message} ${actual} is not within ${tolerance} of ${expected}`.trim());

// The cdf the fake model gives for the context that predicted a step, found by replaying the tokens.
const cdfOf = async (zip, tokens, index, overlap) => {
    const model = zip.model;
    const windows = [];
    // A window holds up to nCtx tokens: BOS, the overlap, then new tokens until the context is full.
    let position = 0;
    while (position < tokens.length) {
        const from = Math.max(0, position - overlap);
        const prefixLength = 1 + (position - from);
        const count = Math.min(tokens.length - position, model.nCtx - prefixLength + 1);
        windows.push({ from, position, count });
        position += count;
    }
    const window = windows.find(w => index >= w.position && index < w.position + w.count);
    model.reset();
    model.eval([model.tokenBos(), ...tokens.slice(window.from, index)]);
    return computeCdf(model.logits());
};

// selectTopTokens

test('selectTopTokens matches a full sort with ties going to the lower id', async () => {
    const frequencies = [5, 9, 9, 1, 9, 3, 7, 7, 2, 9, 1];
    const cdf = [];
    let sum = 0;
    for (const f of frequencies) cdf.push(sum += f);
    const expected = frequencies.map((f, i) => [f, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, i]) => i);
    for (const k of [0, 1, 3, 5, 11, 20]) assert.deepEqual(selectTopTokens(cdf, k), expected.slice(0, k), `k=${k}`);
});

// Step content

test('a step has the documented fields and survives JSON', async () => {
    const zip = makeZip();
    const { steps } = await collect(zip, SHORT);
    assert.ok(steps.length > 3);
    for (const step of steps) {
        assert.deepEqual(Object.keys(step), [
            'index', 'tokenId', 'piece', 'pieceBytes', 'isEos', 'probability', 'bits', 'cumulativeBits', 'lo', 'hi',
            'candidates', 'otherProbability', 'emittedBits', 'newBits', 'outputBits', 'pendingUnderflow', 'coderLow',
            'coderHigh', 'windowIndex', 'windowReset', 'contextLength',
        ]);
        const text = JSON.stringify(step);
        assert.equal(JSON.stringify(JSON.parse(text)), text);
        assert.deepEqual(JSON.parse(text), JSON.parse(JSON.stringify(step)));
        const walk = value => {
            assert.notEqual(typeof value, 'bigint');
            assert.ok(!ArrayBuffer.isView(value), 'typed array in a step');
            if (value && typeof value === 'object') Object.values(value).forEach(walk);
        };
        walk(step);
        assert.ok(/^[0-9a-f]{16}$/.test(step.coderLow) && /^[0-9a-f]{16}$/.test(step.coderHigh));
    }
    assert.deepEqual(steps.map(step => step.index), steps.map((_, i) => i));
    assert.ok(steps[steps.length - 1].isEos);
    assert.equal(steps.filter(step => step.isEos).length, 1);
    assert.equal(steps[0].piece, 'the');
});

test('candidate intervals match the cdf and the probabilities add up', async () => {
    const zip = makeZip();
    const tokens = Array.from(zip.model.tokenize(SHORT, { addBos: false }));
    tokens.push(zip.model.tokenEos());
    const { steps } = await collect(zip, SHORT, 0, { topK: 5 });
    for (const step of steps) {
        const cdf = await cdfOf(zip, tokens, step.index, 0);
        const total = cdf[cdf.length - 1];
        const at = id => [(id > 0 ? cdf[id - 1] : 0) / total, cdf[id] / total];
        assert.equal(step.tokenId, tokens[step.index]);
        assert.deepEqual([step.lo, step.hi], at(step.tokenId));
        assert.equal(step.probability, step.hi - step.lo === 0 ? 0 : (cdf[step.tokenId] - (step.tokenId > 0 ? cdf[step.tokenId - 1] : 0)) / total);
        near(step.bits, -Math.log2(step.probability), 1e-12);
        let sum = step.otherProbability;
        for (const candidate of step.candidates) {
            assert.deepEqual([candidate.lo, candidate.hi], at(candidate.tokenId));
            assert.equal(candidate.chosen, candidate.tokenId === step.tokenId);
            near(candidate.p, candidate.hi - candidate.lo, 1e-15);
            sum += candidate.p;
        }
        near(sum, 1, 1e-12, 'probabilities');
        const ps = step.candidates.map(candidate => candidate.p);
        assert.deepEqual(ps, [...ps].sort((a, b) => b - a));
        assert.equal(step.candidates.filter(candidate => candidate.chosen).length, 1);
        assert.ok(step.candidates.length === 5 || step.candidates.length === 6);
        // The listed tokens are really the most probable ones.
        const listed = new Set(step.candidates.map(candidate => candidate.tokenId));
        const smallest = Math.min(...step.candidates.filter(candidate => !candidate.chosen).map(candidate => candidate.p));
        for (let id = 0; id < cdf.length; id++) {
            const p = (cdf[id] - (id > 0 ? cdf[id - 1] : 0)) / total;
            if (!listed.has(id)) assert.ok(p <= smallest, `token ${id} is more probable than a listed one`);
        }
    }
});

test('the chosen token is listed even when it is not in the top K', async () => {
    const zip = makeZip();
    const { steps } = await collect(zip, TEXT, 0, { topK: 1 });
    const outside = steps.filter(step => step.candidates.length === 2);
    assert.ok(outside.length > 0, 'expected some steps with the chosen token outside the top 1');
    for (const step of outside) assert.ok(step.candidates.some(candidate => candidate.chosen && candidate.tokenId === step.tokenId));
    const none = await collect(makeZip(), SHORT, 0, { topK: 0 });
    for (const step of none.steps) {
        assert.equal(step.candidates.length, 1);
        near(step.otherProbability, 1 - step.probability, 1e-12);
    }
});

test('piece decodes leniently and keeps the bytes of a partial character', async () => {
    const zip = makeZip();
    const { steps } = await collect(zip, bytesOf('héllo €'));
    const split = steps.find(step => step.pieceBytes !== null);
    assert.ok(split, 'the fake model has one token per byte, so a multibyte character is split');
    assert.ok(split.piece.includes('�'));
    assert.ok(split.pieceBytes.length >= 1 && split.pieceBytes.every(byte => Number.isInteger(byte) && byte >= 0 && byte < 256));
    for (const step of steps.filter(step => step.pieceBytes === null)) assert.ok(!step.piece.includes('�'));
});

test('the chosen intervals nest, and the coder interval stays inside the 64-bit range', async () => {
    const { steps } = await collect(makeZip(), TEXT);
    for (const step of steps) {
        assert.ok(step.lo >= 0 && step.lo < step.hi && step.hi <= 1);
        assert.ok(BigInt(`0x${step.coderLow}`) <= BigInt(`0x${step.coderHigh}`));
    }
    const bits = steps.reduce((sum, step) => sum - Math.log2(step.probability), 0);
    near(bits, steps[steps.length - 1].cumulativeBits, 1e-9);
});

test('cumulativeBits and emitted bits agree with the compressed size', async () => {
    const zip = makeZip();
    const { steps, compressed } = await collect(zip, TEXT);
    let running = 0;
    for (const step of steps) {
        running += step.bits;
        near(step.cumulativeBits, running, 1e-9);
    }
    const total = steps[steps.length - 1].cumulativeBits;
    // The coder holds back at most 64 bits of precision plus underflow bits, and finish adds one.
    assert.ok(compressed.length * 8 >= Math.floor(total), `${compressed.length * 8} bits for ${total}`);
    assert.ok(compressed.length * 8 <= Math.ceil(total) + 66 + 8, `${compressed.length * 8} bits for ${total}`);
    let previous = 0;
    let text = '';
    for (const step of steps) {
        assert.ok(step.emittedBits >= previous);
        assert.equal(step.newBits.length, step.emittedBits - previous);
        text += step.newBits;
        assert.equal(step.outputBits, text.slice(-256));
        previous = step.emittedBits;
    }
    const last = steps[steps.length - 1];
    // Every emitted bit is a prefix of the compressed stream.
    const stream = Array.from(compressed, byte => byte.toString(2).padStart(8, '0')).join('');
    assert.equal(stream.slice(0, last.emittedBits), text);
    assert.ok(stream.length - last.emittedBits <= last.pendingUnderflow + 65);
});

test('window changes are reported', async () => {
    const zip = makeZip();
    const { steps } = await collect(zip, TEXT, 8);
    const resets = steps.filter(step => step.windowReset);
    assert.ok(resets.length >= 1);
    assert.equal(steps[0].windowIndex, 0);
    assert.equal(steps[0].windowReset, false);
    for (const step of resets) assert.equal(steps[step.index - 1].windowIndex, step.windowIndex - 1);
    assert.equal(steps[steps.length - 1].windowIndex, resets.length);
    for (const step of steps) assert.ok(step.contextLength >= 1 && step.contextLength <= 64);
});

// Same output, with and without observers

for (const overlap of [0, 8]) {
    test(`output is identical with and without onStep (overlap ${overlap})`, async () => {
        const plain = await makeZip().compress(TEXT, overlap);
        const observed = await collect(makeZip(), TEXT, overlap);
        const traced = makeZip();
        const generator = traced.trace(TEXT, overlap);
        let result;
        for (;;) {
            const next = await generator.next();
            if (next.done) {
                result = next.value;
                break;
            }
        }
        assert.equal(toHex(observed.compressed), toHex(plain));
        assert.equal(toHex(result.compressed), toHex(plain));
        assert.equal(result.tokens, observed.steps.length);
        near(result.totalBits, observed.steps[observed.steps.length - 1].cumulativeBits, 0);
    });
}

test('decode steps equal encode steps', async () => {
    for (const overlap of [0, 8]) {
        const encoded = await collect(makeZip(), TEXT, overlap);
        const decodeSteps = [];
        const original = await makeZip().decompress(encoded.compressed, overlap, { onStep: step => void decodeSteps.push(step) });
        assert.equal(toHex(original), toHex(TEXT));
        assert.equal(JSON.stringify(decodeSteps), JSON.stringify(encoded.steps));
        const quiet = await makeZip().decompress(encoded.compressed, overlap);
        assert.equal(toHex(quiet), toHex(original));
    }
});

test('onStep may be async and is awaited', async () => {
    const zip = makeZip();
    const log = [];
    let running = false;
    await zip.compress(SHORT, 0, {
        onStep: async step => {
            assert.ok(!running, 'a step arrived before the previous callback finished');
            running = true;
            await new Promise(resolve => setTimeout(resolve, 1));
            log.push(step.index);
            running = false;
        },
    });
    assert.deepEqual(log, log.map((_, i) => i));
    assert.ok(log.length > 3);
});

test('an error in onStep rejects and leaves the model usable', async () => {
    const zip = makeZip();
    await assert.rejects(() => zip.compress(SHORT, 0, { onStep: step => { if (step.index === 2) throw new Error('stop here'); } }), /stop here/);
    const again = await zip.compress(SHORT, 0);
    assert.equal(toHex(again), toHex(await makeZip().compress(SHORT, 0)));
});

// trace()

test('trace can be abandoned early without leaking state', async () => {
    const zip = makeZip();
    const generator = zip.trace(TEXT, 0);
    const first = await generator.next();
    assert.equal(first.value.index, 0);
    await generator.return();
    assert.equal(zip.busy, false);
    assert.ok(zip.model.context.length === 0, 'the half-finished window is cleared');
    const compressed = await zip.compress(TEXT, 0);
    assert.equal(toHex(compressed), toHex(await makeZip().compress(TEXT, 0)));
    // break out of for await does the same
    for await (const step of zip.trace(TEXT, 0)) if (step.index === 3) break;
    assert.equal(zip.busy, false);
});

test('a second call on a busy LlamaZip is refused', async () => {
    const zip = makeZip();
    const generator = zip.trace(SHORT, 0);
    await generator.next();
    await assert.rejects(() => zip.compress(SHORT, 0), /already compressing/);
    await generator.return();
    await zip.compress(SHORT, 0);
});

test('trace takes the abort signal like compress', async () => {
    const zip = makeZip();
    const controller = new AbortController();
    const steps = [];
    const generator = zip.trace(TEXT, 0, { signal: controller.signal });
    let result;
    for (;;) {
        const next = await generator.next();
        if (next.done) {
            result = next.value;
            break;
        }
        steps.push(next.value);
        if (steps.length === 3) controller.abort();
    }
    assert.equal(steps.length, 4);
    assert.ok(steps[3].isEos);
    const text = await makeZip().decompress(result.compressed, 0);
    assert.equal(new TextDecoder().decode(text), steps.slice(0, 3).map(step => step.piece).join(''));
});

test('topK must be a non-negative integer', async () => {
    await assert.rejects(() => collect(makeZip(), SHORT, 0, { topK: -1 }), /topK/);
    await assert.rejects(() => collect(makeZip(), SHORT, 0, { topK: 1.5 }), /topK/);
});

// CLI

test('--trace prints one JSON line per step instead of the compressed data', async () => {
    const io = captureIo();
    const code = await main(['fake.gguf', '--trace', '-c', 'the thing'], {
        io,
        readStdin: async () => new Uint8Array(0),
        createZip: (path, options) => LlamaZip.load(path, { ...options, createModel }),
        onSigint: () => () => {},
    });
    assert.equal(code, 0);
    const text = new TextDecoder().decode(io.outBytes());
    assert.ok(text.endsWith('\n'));
    const lines = text.trimEnd().split('\n').map(line => JSON.parse(line));
    const { steps } = await collect(makeZip(), SHORT);
    assert.equal(JSON.stringify(lines), JSON.stringify(steps));
});

test('--trace needs -c', async () => {
    const io = captureIo();
    const code = await main(['fake.gguf', '--trace', '-d', 'AAAA'], { io, createZip: async () => { throw new Error('no model expected'); } });
    assert.equal(code, 2);
    assert.ok(io.errText().includes('--trace'));
});

// Real model

const MODEL = tjs.env.LLAMA_ZIP_TEST_MODEL ?? '/tmp/claude-1000/-home-kaki-Documents-git-github-com-VibedByKaKi-llmzip-desktop/'
    + '4eff2479-8a67-41fd-9301-2500fcb1ab44/scratchpad/models/SmolLM2-135M.Q8_0.gguf';
let real = null;
let reason = null;
try {
    await tjs.stat(MODEL);
    real = new LlamaZip(MODEL, { nCtx: 128, nGpuLayers: 0 });
}
catch (error) {
    reason = String(error?.message ?? error).split('\n')[0].slice(0, 120);
}

if (real) {
    plainTest('real model: steps are consistent and the output does not change', async () => {
        const data = bytesOf('We the People of the United States');
        const plain = await real.compress(data, 0);
        const steps = [];
        const start = Date.now();
        const observed = await real.compress(data, 0, { onStep: step => void steps.push(step) });
        info(`${steps.length} tokens in ${Date.now() - start} ms with onStep`);
        assert.equal(toHex(observed), toHex(plain));
        assert.equal(JSON.stringify(JSON.parse(JSON.stringify(steps))), JSON.stringify(steps));
        let sum = 0;
        for (const step of steps) {
            sum += step.probability;
            let listed = step.otherProbability;
            for (const candidate of step.candidates) listed += candidate.p;
            near(listed, 1, 1e-9);
            assert.equal(step.candidates.length >= 16, true);
            assert.ok(step.candidates.some(candidate => candidate.chosen));
        }
        assert.ok(sum > 0);
        assert.ok(steps[steps.length - 1].isEos);
        near(steps[steps.length - 1].cumulativeBits, observed.length * 8, 70);
        const decoded = [];
        const text = await real.decompress(observed, 0, { onStep: step => void decoded.push(step) });
        assert.equal(new TextDecoder().decode(text), 'We the People of the United States');
        assert.equal(JSON.stringify(decoded), JSON.stringify(steps));
    });
}
else {
    skip('real model: steps are consistent and the output does not change', reason);
}

await run();
