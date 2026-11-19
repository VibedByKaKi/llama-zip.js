// Port of tests/python/test_compute_cdf.py. Run with `tjs run js/test/compute_cdf.test.js`.
import { FREQ_SCALE_FACTOR, LlamaZip, computeCdf, roundHalfEven } from '../llamazip.js';
import { assert, run, test } from './run.js';
import { fakeLogits, createModel } from './fake_llama.js';
import { loadVectors, makeRng } from './vectors.js';

const VECTORS = await loadVectors('compute_cdf.json');

const zipper = new LlamaZip('fake', { nCtx: 32, createModel });
const f32 = values => Float32Array.from(values);
const cdfOf = values => Array.from(zipper.computeCdf(f32(values)));
const freqsOf = cdf => cdf.map((value, i) => value - (i ? cdf[i - 1] : 0));

for (const c of VECTORS.cases) {
    test(`computeCdf vectors [${c.name}]`, () => {
        assert.deepEqual(cdfOf(c.logits), c.cum_freqs);
    });
}

for (const c of VECTORS.cases) {
    test(`cdf is strictly increasing with floor of one [${c.name}]`, () => {
        const cdf = cdfOf(c.logits);
        assert.ok(freqsOf(cdf).every(freq => freq >= 1));
        assert.equal(cdf.length, c.logits.length);
    });
}

for (const c of VECTORS.cases) {
    test(`cdf total is scale factor plus floor bumps [${c.name}]`, () => {
        const cdf = cdfOf(c.logits);
        const n = cdf.length;
        assert.ok(FREQ_SCALE_FACTOR - n <= cdf[n - 1] && cdf[n - 1] <= FREQ_SCALE_FACTOR + n);
    });
}

test('scale factor constant in vectors', () => {
    assert.equal(VECTORS.scale_factor, FREQ_SCALE_FACTOR);
    assert.equal(FREQ_SCALE_FACTOR, 2 ** 32);
});

test('uniform logits split evenly', () => {
    assert.deepEqual(cdfOf(new Array(8).fill(0)), Array.from({ length: 8 }, (_, i) => (i + 1) * 2 ** 29));
});

test('zero probability tokens get frequency one', () => {
    assert.deepEqual(freqsOf(cdfOf([0, -200, -200])).slice(1), [1, 1]);
});

test('cdf is invariant to a constant shift', () => {
    assert.deepEqual(cdfOf([1, 2, 3]), cdfOf([51, 52, 53]));
});

test('cdf is float64', () => {
    assert.ok(zipper.computeCdf(fakeLogits([0, 4])) instanceof Float64Array);
});

test('cdf tracks the logits order', () => {
    const freqs = freqsOf(cdfOf([0, 3, 1]));
    assert.ok(freqs[1] > freqs[2] && freqs[2] > freqs[0]);
});

test('cdf for full fake vocab is monotonic', () => {
    for (const context of [[0], [0, 5], [0, 5, 9, 12]]) {
        assert.ok(freqsOf(Array.from(zipper.computeCdf(fakeLogits(context)))).every(freq => freq >= 1));
    }
});

// The Python test spies on the model's logits_to_logprobs. This port does the softmax itself, so
// the equivalent check is that computeCdf is a pure function of the logits.
test('computeCdf does not modify its input and matches the exported function', () => {
    const logits = f32([0, 3, 1]);
    const before = Array.from(logits);
    assert.deepEqual(zipper.computeCdf(logits), computeCdf(logits));
    assert.deepEqual(logits, before);
});

// Additions for the JS port: Math.round rounds halves up, numpy.round to even.
test('roundHalfEven matches numpy.round', () => {
    const cases = [[0.5, 0], [1.5, 2], [2.5, 2], [3.5, 4], [0.4999999, 0], [0.5000001, 1], [4294967296.5, 4294967296], [4294967297.5, 4294967298], [7, 7], [0, 0]];
    for (const [input, expected] of cases) assert.equal(roundHalfEven(input), expected, `round(${input})`);
});

test('a logit of minus infinity gets the floor frequency', () => {
    const cdf = Array.from(computeCdf(f32([0, -Infinity])));
    assert.equal(cdf[0], FREQ_SCALE_FACTOR);
    assert.equal(cdf[1], FREQ_SCALE_FACTOR + 1);
});

// The optimized computeCdf skips exp() for tokens at the floor and sums in numpy's order. Check it
// against a plain float64 softmax on random logits, including a vocabulary-sized one.
test('computeCdf agrees with a naive softmax to within one unit per token', () => {
    const rng = makeRng(99);
    for (const size of [5, 300, 2000, 49152, 70000]) {
        const logits = Float32Array.from({ length: size }, () => (rng.below(100000) / 100000 - 0.5) * (size > 1000 ? 24 : 6));
        const max = logits.reduce((a, b) => Math.max(a, b), -Infinity);
        const exps = Array.from(logits, value => Math.exp(value - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        const cdf = computeCdf(logits);
        for (let i = 0; i < size; i++) {
            const expected = Math.max(1, roundHalfEven(FREQ_SCALE_FACTOR * (exps[i] / sum)));
            const actual = cdf[i] - (i ? cdf[i - 1] : 0);
            assert.ok(Math.abs(actual - expected) <= 1, `size ${size} index ${i}: ${actual} vs ${expected}`);
        }
    }
});

await run();
