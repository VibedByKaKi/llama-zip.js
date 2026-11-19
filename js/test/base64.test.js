// Port of tests/python/test_base64.py. Run with `tjs run js/test/base64.test.js`.
import { BASE64, BASE64_EQ, b64encode, robustB64decode } from '../cli.js';
import { assert, run, test } from './run.js';
import { bytesOf, fromHex, loadVectors, toHex } from './vectors.js';

const VECTORS = await loadVectors('base64.json');

for (const c of VECTORS.cases) {
    test(`robustB64decode vectors [${c.name}]`, () => {
        assert.equal(toHex(robustB64decode(fromHex(c.input_hex))), c.output_hex);
    });
}

test('alphabet constants', () => {
    assert.equal(BASE64.length, 64);
    assert.equal(new Set(BASE64).size, 64);
    assert.equal(BASE64_EQ, BASE64 + '=');
    assert.equal(BASE64, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/');
});

test('padding is replaced by zero bits', () => {
    // "aGVsbG8=" is "hello", but the "=" is dropped and the length is padded
    // with "A" (zero bits), so one extra zero byte follows. It is harmless for
    // arithmetic coded data because the decoder reads zeros past the end.
    assert.deepEqual(robustB64decode(bytesOf('aGVsbG8=')), [...bytesOf('hello'), 0]);
});

test('valid unpadded multiple of four is exact', () => {
    assert.deepEqual(robustB64decode(bytesOf('aGVsbG8h')), bytesOf('hello!'));
});

test('whitespace and garbage are ignored', () => {
    assert.deepEqual(robustB64decode(bytesOf(' a G\tV\ns\r\nb G 8 h ')), bytesOf('hello!'));
    const noisy = Uint8Array.from([0xff, ...bytesOf('aGVs'), 0, ...bytesOf('bG8h'), 0xfe]);
    assert.deepEqual(robustB64decode(noisy), bytesOf('hello!'));
});

test('empty and all invalid input give empty output', () => {
    assert.equal(robustB64decode(new Uint8Array(0)).length, 0);
    assert.equal(robustB64decode(bytesOf('=-_!? \n')).length, 0);
});

test('urlsafe characters are dropped, not translated', () => {
    assert.equal(robustB64decode(bytesOf('-_-_')).length, 0);
});

for (let size = 0; size < 40; size++) {
    test(`round trip with trailing zero allowance [${size}]`, () => {
        const data = Uint8Array.from({ length: size }, (_, i) => (i * 37 + 5) % 256);
        const decoded = robustB64decode(bytesOf(b64encode(data)));
        assert.equal(toHex(decoded.slice(0, size)), toHex(data));
        assert.ok(decoded.slice(size).every(byte => byte === 0));
        assert.ok(decoded.length - size <= 2);
    });
}

test('accepts bytes from a plain array', () => {
    assert.deepEqual(robustB64decode(Array.from(bytesOf('aGVsbG8h'))), bytesOf('hello!'));
});

test('module has no stray alphabet difference', () => {
    assert.equal(BASE64, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/');
});

// b64encode is new in the port, because JavaScript has no binary-safe built in. Compare with known values.
test('b64encode matches known encodings', () => {
    assert.equal(b64encode(new Uint8Array(0)), '');
    assert.equal(b64encode(bytesOf('f')), 'Zg==');
    assert.equal(b64encode(bytesOf('fo')), 'Zm8=');
    assert.equal(b64encode(bytesOf('foo')), 'Zm9v');
    assert.equal(b64encode(bytesOf('hello')), 'aGVsbG8=');
    assert.equal(b64encode(Uint8Array.of(0xff, 0xfe, 0xfd)), '//79');
});

await run();
