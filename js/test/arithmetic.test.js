// Port of tests/python/test_arithmetic.py. Run with `tjs run js/test/arithmetic.test.js`.
import { ArithmeticCoderBase, Decoder, Encoder, NUM_STATE_BITS } from '../arithmetic.js';
import { FREQ_SCALE_FACTOR } from '../llamazip.js';
import { assert, run, test } from './run.js';
import { fromHex, loadVectors, makeRng, toHex } from './vectors.js';

const VECTORS = await loadVectors('arithmetic.json');

const encode = (cdfs, steps) => {
    const encoder = new Encoder();
    for (const [cdfIndex, symbol] of steps) encoder.encodeSymbol(cdfs[cdfIndex], symbol);
    encoder.finish();
    return encoder.getEncoded();
};

const decodeAll = (decoder, cdfs, steps) => steps.map(([cdfIndex]) => decoder.decodeSymbol(cdfs[cdfIndex]));

for (const c of VECTORS.cases) {
    test(`encode matches vector [${c.name}]`, () => {
        assert.equal(toHex(encode(c.cdfs, c.steps)), c.encoded_hex);
    });
}

for (const c of VECTORS.cases) {
    test(`decode matches vector [${c.name}]`, () => {
        const decoder = new Decoder(fromHex(c.encoded_hex));
        assert.deepEqual(decodeAll(decoder, c.cdfs, c.steps), c.steps.map(([, symbol]) => symbol));
    });
}

// The Python test passes numpy float64 arrays, the equivalent here is Float64Array.
for (const c of VECTORS.cases) {
    test(`decode accepts typed array cdf [${c.name}]`, () => {
        const decoder = new Decoder(fromHex(c.encoded_hex));
        const cdfs = c.cdfs.map(cdf => Float64Array.from(cdf));
        assert.deepEqual(decodeAll(decoder, cdfs, c.steps), c.steps.map(([, symbol]) => symbol));
    });
}

for (const c of VECTORS.decode_only) {
    test(`decode arbitrary data [${c.name}]`, () => {
        const decoder = new Decoder(fromHex(c.data_hex));
        assert.deepEqual(c.symbols.map(() => decoder.decodeSymbol(c.cdf)), c.symbols);
    });
}

test('underflow vectors exercise underflow', () => {
    const byName = Object.fromEntries(VECTORS.cases.map(c => [c.name, c]));
    assert.equal(byName['underflow middle symbol repeated'].max_underflow, 47);
    assert.ok(VECTORS.cases.filter(c => c.max_underflow > 0).length >= 8);
});

test('empty message is one set bit', () => {
    assert.deepEqual(encode([[1, 2]], []), [0x80]);
});

test('single symbol alphabet costs one bit whatever the length', () => {
    assert.deepEqual(encode([[1]], Array.from({ length: 1000 }, () => [0, 0])), [0x80]);
});

test('binary alphabet hand computed', () => {
    // 0 -> low half (bit 0), 1 -> high half (bit 1); finish() adds a trailing 1.
    assert.deepEqual(encode([[1, 2]], [[0, 0]]), [0x40]);
    assert.deepEqual(encode([[1, 2]], [[0, 1]]), [0xc0]);
    assert.deepEqual(encode([[1, 2]], [[0, 0], [0, 1], [0, 0], [0, 1]]), [0b01011000]);
});

test('uniform 256 writes the symbols as bytes', () => {
    const cdf = Array.from({ length: 256 }, (_, i) => i + 1);
    const data = encode([cdf], Array.from({ length: 256 }, (_, b) => [0, b]));
    assert.deepEqual(data.slice(0, 256), Array.from({ length: 256 }, (_, i) => i));
});

test('initial state', () => {
    const coder = new ArithmeticCoderBase();
    assert.equal(coder.low, 0n);
    assert.equal(coder.high, (1n << 64n) - 1n);
    assert.equal(coder.halfRange, 1n << 63n);
    assert.equal(coder.quarterRange, 1n << 62n);
    assert.equal(coder.stateMask, (1n << 64n) - 1n);
});

test('base class hooks are abstract', () => {
    const coder = new ArithmeticCoderBase();
    assert.throws(() => coder.shift(), /not implemented/);
    assert.throws(() => coder.underflow(), /not implemented/);
});

test('state stays in 64 bits', () => {
    const rng = makeRng(7);
    const encoder = new Encoder();
    const cdf = [1, 5, 2 ** 32, 2 ** 32 + 3];
    for (let i = 0; i < 500; i++) {
        encoder.encodeSymbol(cdf, rng.below(4));
        assert.ok(encoder.low >= 0n && encoder.low <= encoder.high && encoder.high <= (1n << 64n) - 1n);
        assert.ok(encoder.high - encoder.low >= encoder.quarterRange);
    }
});

test('encoder underflow counter is flushed on shift', () => {
    const encoder = new Encoder();
    for (let i = 0; i < 5; i++) encoder.encodeSymbol([1, 2, 3], 1);
    assert.ok(encoder.numUnderflow >= 5);
    const pending = encoder.numUnderflow;
    assert.equal(encoder.getEncoded().length, 0);
    encoder.encodeSymbol([1, 2, 3], 0);
    assert.equal(encoder.numUnderflow, 0);
    assert.ok(encoder.getEncoded().length * 8 >= pending);
});

test('encoder bit packing', () => {
    const encoder = new Encoder();
    for (const bit of [1, 0, 1, 1, 0, 0, 0, 1, 1]) encoder.appendBit(bit);
    assert.deepEqual(encoder.getEncoded(), [0b10110001, 0b10000000]);
});

test('encoder output grows past its first buffer', () => {
    const encoder = new Encoder();
    for (let i = 0; i < 1000; i++) encoder.appendBit(1);
    assert.equal(encoder.getEncoded().length, 125);
});

test('decoder reads zero bits past the end', () => {
    let decoder = new Decoder(new Uint8Array(0));
    assert.equal(decoder.code, 0n);
    assert.equal(decoder.readCodeBit(), 0);
    decoder = new Decoder(Uint8Array.of(0xff));
    assert.equal(decoder.code, 0xFFn << 56n);
});

test('decoder reads msb first', () => {
    const decoder = new Decoder(Uint8Array.from([0xa5, 0, 0, 0, 0, 0, 0, 0, 0x80]));
    assert.equal(decoder.code, 0xA5n << 56n);
    assert.equal(decoder.readCodeBit(), 1);
    assert.equal(decoder.readCodeBit(), 0);
});

test('decoder counts the bits it reads, including zeros past the end', () => {
    const decoder = new Decoder(Uint8Array.of(0xff));
    assert.equal(decoder.bitsRead, 64);
    decoder.readCodeBit();
    assert.equal(decoder.bitsRead, 65);
});

test('trailing zero bytes do not change the decoded symbols', () => {
    const c = VECTORS.cases[VECTORS.cases.length - 1];
    const data = new Uint8Array([...fromHex(c.encoded_hex), ...new Uint8Array(9)]);
    assert.deepEqual(decodeAll(new Decoder(data), c.cdfs, c.steps), c.steps.map(([, symbol]) => symbol));
});

test('random round trips', () => {
    const rng = makeRng(2024);
    for (let round = 0; round < 200; round++) {
        const size = rng.between(1, 40);
        let total = rng.choice([size, size * 3, 1000, 1 << 16, 2 ** 32, 2 ** 32 + size]);
        total = Math.max(total, size);
        // size - 1 distinct cut points in [1, total)
        const cuts = new Set();
        while (cuts.size < size - 1) cuts.add(rng.between(1, total - 1));
        const cdf = [...[...cuts].sort((a, b) => a - b), total];
        const steps = Array.from({ length: rng.between(0, 150) }, () => [0, rng.below(size)]);
        const decoder = new Decoder(encode([cdf], steps));
        assert.deepEqual(steps.map(() => decoder.decodeSymbol(cdf)), steps.map(([, symbol]) => symbol));
    }
});

test('compressed size tracks information content', () => {
    // 4096 symbols at 8 bits each fill 4096 bytes, and finish() starts one more.
    const cdf = Array.from({ length: 256 }, (_, i) => i + 1);
    const data = encode([cdf], Array.from({ length: 4096 }, (_, i) => [0, i % 256]));
    assert.equal(data.length, 4097);
});

test('module constants', () => {
    assert.equal(NUM_STATE_BITS, 64);
    assert.equal(FREQ_SCALE_FACTOR, 2 ** 32);
});

await run();
