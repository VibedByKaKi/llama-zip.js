// Port of tests/python/test_utf8.py. Run with `tjs run js/test/utf8.test.js`.
import { PUA_START, Utf8Chunk, Utf8Chunks, bytesToUtf8, utf8ToBytes } from '../utf8.js';
import { assert, run, test } from './run.js';
import { bytesOf, concatBytes, fromHex, loadVectors, makeRng, toHex } from './vectors.js';

const VECTORS = await loadVectors('utf8.json');

const strictDecode = bytes => new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
const chr = codePoint => String.fromCodePoint(codePoint);
const puaText = bytes => bytes.map(b => chr(PUA_START + b)).join('');
const chunksOf = data => Array.from(new Utf8Chunks(data), chunk => [chunk.valid, chunk.invalid]);

for (const c of VECTORS.chunks) {
    test(`utf8 chunks vectors [${c.name}]`, () => {
        const expected = c.chunks.map(chunk => [toHex(fromHex(chunk.valid_hex)), toHex(fromHex(chunk.invalid_hex))]);
        assert.deepEqual(chunksOf(fromHex(c.input_hex)).map(([valid, invalid]) => [toHex(valid), toHex(invalid)]), expected);
    });
}

for (const c of VECTORS.chunks) {
    test(`utf8 chunks cover the input exactly [${c.name}]`, () => {
        const data = fromHex(c.input_hex);
        assert.equal(toHex(concatBytes(...chunksOf(data).flat())), toHex(data));
    });
}

for (const c of VECTORS.chunks) {
    test(`utf8 chunks valid part decodes [${c.name}]`, () => {
        for (const [valid] of chunksOf(fromHex(c.input_hex))) strictDecode(valid);
    });
}

test('utf8 chunks empty input yields nothing', () => {
    assert.equal(Array.from(new Utf8Chunks(new Uint8Array(0))).length, 0);
});

test('utf8 chunks is its own iterator', () => {
    const chunks = new Utf8Chunks(Uint8Array.of(0x61, 0xff));
    assert.ok(chunks[Symbol.iterator]() === chunks);
    assert.ok(chunks.next().value instanceof Utf8Chunk);
});

test('utf8 chunks matches TextDecoder for valid text', () => {
    const text = `aé€${chr(0x1f600)}${chr(0x10ffff)}퟿ end`;
    const result = chunksOf(bytesOf(text));
    assert.equal(result.length, 1);
    assert.equal(toHex(result[0][0]), toHex(bytesOf(text)));
    assert.equal(result[0][1].length, 0);
});

test('utf8 chunks valid prefix agrees with the TextDecoder', () => {
    const rng = makeRng(5);
    const alphabet = [0x41, 0x80, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98, 0x80, 0xFF, 0xED, 0xA0];
    for (let round = 0; round < 500; round++) {
        const data = Uint8Array.from({ length: rng.between(0, 12) }, () => rng.choice(alphabet));
        const parts = [];
        for (const [valid, invalid] of chunksOf(data)) {
            strictDecode(valid);
            // The invalid run must really be undecodable as a lone chunk.
            if (invalid.length) assert.throws(() => strictDecode(invalid), /decoding error/);
            parts.push(valid, invalid);
        }
        assert.equal(toHex(concatBytes(...parts)), toHex(data));
        strictDecode(bytesToUtf8(data));
    }
});

for (const c of VECTORS.bytes_to_utf8) {
    test(`bytes_to_utf8 vectors [${c.name}]`, () => {
        const result = bytesToUtf8(fromHex(c.input_hex));
        assert.equal(toHex(result), c.output_hex);
        strictDecode(result);
    });
}

test('bytes_to_utf8 passes ordinary text through', () => {
    const text = `plain text é € ${chr(0x1f600)}`;
    assert.equal(toHex(bytesToUtf8(bytesOf(text))), toHex(bytesOf(text)));
});

test('bytes_to_utf8 maps invalid bytes into the PUA', () => {
    assert.equal(strictDecode(bytesToUtf8(Uint8Array.of(0xff))), chr(PUA_START + 0xFF));
    assert.equal(strictDecode(bytesToUtf8(Uint8Array.of(0x80))), chr(PUA_START + 0x80));
    assert.equal(strictDecode(bytesToUtf8(Uint8Array.of(0x61, 0xc3))), 'a' + chr(PUA_START + 0xC3));
});

test('bytes_to_utf8 escapes real PUA characters byte by byte', () => {
    // U+E000 is EE 80 80 in UTF-8, so it must come back as three PUA characters.
    assert.equal(strictDecode(bytesToUtf8(bytesOf(''))), puaText([0xEE, 0x80, 0x80]));
    // U+E100 is outside the mapped range and stays as is.
    assert.equal(toHex(bytesToUtf8(bytesOf(''))), toHex(bytesOf('')));
});

test('PUA boundaries', () => {
    assert.notEqual(toHex(bytesToUtf8(bytesOf(''))), toHex(bytesOf('')));
    assert.equal(toHex(bytesToUtf8(bytesOf(''))), toHex(bytesOf('')));
    // U+DFFF as UTF-8 (surrogatepass) is ED BF BF, which is not valid UTF-8, so every byte is escaped.
    assert.equal(toHex(bytesToUtf8(Uint8Array.of(0xED, 0xBF, 0xBF))), toHex(bytesOf(puaText([0xED, 0xBF, 0xBF]))));
});

for (const c of VECTORS.utf8_to_bytes) {
    test(`utf8_to_bytes vectors [${c.name}]`, () => {
        assert.equal(toHex(utf8ToBytes(c.codepoints.map(chr).join(''))), c.output_hex);
    });
}

test('utf8_to_bytes maps PUA to single bytes', () => {
    assert.deepEqual(utf8ToBytes(chr(PUA_START)), [0]);
    assert.deepEqual(utf8ToBytes(chr(PUA_START + 0xFF)), [0xff]);
    assert.equal(toHex(utf8ToBytes('')), toHex(bytesOf('')));
});

test('utf8_to_bytes empty', () => {
    assert.equal(utf8ToBytes('').length, 0);
});

for (const c of VECTORS.roundtrip) {
    test(`round trip vectors [${c.name}]`, () => {
        const data = fromHex(c.input_hex);
        assert.equal(toHex(utf8ToBytes(strictDecode(bytesToUtf8(data)))), toHex(data));
    });
}

const roundTrips = data => toHex(utf8ToBytes(strictDecode(bytesToUtf8(data)))) === toHex(data);

test('round trip every single byte', () => {
    for (let value = 0; value < 256; value++) assert.ok(roundTrips(Uint8Array.of(value)), `byte ${value}`);
});

test('round trip every pair of bytes sampled', () => {
    const rng = makeRng(11);
    for (let round = 0; round < 3000; round++) assert.ok(roundTrips(Uint8Array.of(rng.below(256), rng.below(256))));
});

test('round trip random bytes', () => {
    const rng = makeRng(42);
    for (let round = 0; round < 300; round++) {
        assert.ok(roundTrips(Uint8Array.from({ length: rng.between(0, 200) }, () => rng.below(256))));
    }
});

test('round trip random text with PUA', () => {
    const rng = makeRng(43);
    const alphabet = ['a', 'é', '', '', '', chr(0x1f600), '€'];
    for (let round = 0; round < 100; round++) {
        const text = Array.from({ length: rng.between(0, 30) }, () => rng.choice(alphabet)).join('');
        assert.ok(roundTrips(bytesOf(text)));
    }
});

test('bytes_to_utf8 is injective on PUA collisions', () => {
    assert.notEqual(toHex(bytesToUtf8(Uint8Array.of(0xff))), toHex(bytesToUtf8(bytesOf(''))));
});

await run();
