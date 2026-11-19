// Optional tests with the LFM3 model (arch lfm3, a hybrid of short convolutions and attention, so
// part of its context is a recurrent state that llama_memory_clear must reset).
//   LLAMA_ZIP_TEST_LFM3=1 tjs run js/test/lfm3.test.js
// Env: LLAMA_ZIP_TEST_LFM3_MODEL (GGUF path, default below). Without LLAMA_ZIP_TEST_LFM3=1 the tests are skipped.
import { LlamaZip } from '../llamazip.js';
import { assert, run, skip, test } from './run.js';
import { bytesOf, concatBytes, toHex } from './vectors.js';

const MODEL = tjs.env.LLAMA_ZIP_TEST_LFM3_MODEL ?? `${tjs.homeDir}/.local/share/app.vibedbykaki.llmzip/models/LFM3-1.7B-Q4_K_M.gguf`;
const same = (actual, expected, message) => assert.equal(toHex(actual), toHex(expected), message);
const bits = values => Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length)).join(',');

if (tjs.env.LLAMA_ZIP_TEST_LFM3 !== '1') skip('LFM3 model', 'set LLAMA_ZIP_TEST_LFM3=1 to run (1.1 GB model)');
else {
    // 256 is the smallest context llama.cpp keeps, so the window test needs only a few hundred tokens.
    const zip = new LlamaZip(MODEL, { nCtx: 256, nGpuLayers: 0 });
    const model = zip.model;
    const TEXT = bytesOf('The quick brown fox jumps over the lazy dog. ');

    test('loads and tokenizes', () => {
        assert.equal(model.nCtx, 256);
        assert.ok(model.nVocab > 1000);
        assert.ok(model.tokenize('Hello world').length >= 2);
    });

    test('reset clears the recurrent state: replays are bit-identical', () => {
        const tokens = [model.tokenBos(), ...model.tokenize('The quick brown fox jumps over the lazy dog.')];
        model.reset();
        model.eval(tokens);
        const first = bits(model.logits());
        model.reset();
        model.eval(model.tokenize('Completely different text about astronomy and the stars.'));
        model.reset();
        model.eval(tokens);
        assert.equal(bits(model.logits()), first, 'batch replay after other content');
        const stepwise = () => {
            model.reset();
            const all = [];
            for (const token of tokens) {
                model.eval([token]);
                all.push(bits(model.logits()));
            }
            return all;
        };
        assert.deepEqual(stepwise(), stepwise(), 'token by token replay');
    });

    test('round trip of a short text', async () => {
        const compressed = await zip.compress(TEXT);
        assert.ok(compressed.length < TEXT.length);
        same(await zip.decompress(compressed), TEXT);
    });

    test('round trip with window overlap (the text is longer than the context)', async () => {
        const data = concatBytes(...Array(30).fill(bytesOf('The quick brown fox jumps over the lazy dog. ')));
        assert.ok(model.tokenize(data).length > model.nCtx);
        const compressed = await zip.compress(data, 32);
        same(await zip.decompress(compressed, 32), data);
    });

    test('round trip of non ASCII and raw bytes', async () => {
        const data = concatBytes(bytesOf('café € \u{1f600} '), Uint8Array.from([0xff, 0xfe, 0x80]));
        same(await zip.decompress(await zip.compress(data)), data);
    });

    test('closing', () => zip.close());
}

await run();
