// Port of tests/python/test_integration.py: round trips with the real libllama and a small GGUF model.
//
// Skipped when libllama cannot be loaded or the model file is missing. Set LLAMA_ZIP_TEST_MODEL to
// use another model and LLAMA_LIB for the path of libllama. Run with `tjs run js/test/integration.test.js`.
//
// The tests after the ported ones run the command line tool as a child process.
import { LlamaZip } from '../llamazip.js';
import { assert, run, skip, test } from './run.js';
import { bytesOf, concatBytes, toHex } from './vectors.js';

const DEFAULT_MODEL = '/tmp/claude-1000/-home-kaki-Documents-git-github-com-VibedByKaKi-llmzip-desktop/'
    + '4eff2479-8a67-41fd-9301-2500fcb1ab44/scratchpad/models/SmolLM2-135M.Q8_0.gguf';
const MODEL = tjs.env.LLAMA_ZIP_TEST_MODEL ?? DEFAULT_MODEL;
const CLI = new URL('../cli.js', import.meta.url).pathname;

const same = (actual, expected, message) => assert.equal(toHex(actual), toHex(expected), message);

const modelExists = async () => {
    try {
        await tjs.stat(MODEL);
        return true;
    }
    catch {
        return false;
    }
};

let zipper = null;
let reason = null;
if (!(await modelExists())) {
    reason = `model file not found: ${MODEL}`;
}
else {
    try {
        zipper = new LlamaZip(MODEL, { nCtx: 128, nGpuLayers: 0 });
    }
    catch (error) {
        reason = `libllama is not usable: ${String(error?.message ?? error).split('\n')[0].slice(0, 120)}`;
    }
}

const TEXT = bytesOf('The quick brown fox jumps over the lazy dog. '.repeat(3));
const EDGE_INPUTS = [
    ['empty', new Uint8Array(0)],
    ['leading space', bytesOf(' leading space')],
    ['non ASCII', bytesOf('café € \u{1f600}')],
    ['raw bytes', Uint8Array.from([0xff, 0xfe, 0x80, ...bytesOf(' raw')])],
];

const readAll = async stream => {
    const reader = stream.getReader();
    const chunks = [];
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    return concatBytes(...chunks);
};

// Runs the CLI as a child process. `after` runs once the process has started, with the process.
const spawnCli = async (args, { stdin = null, after = null } = {}) => {
    const proc = tjs.spawn([tjs.exePath, 'run', CLI, MODEL, ...args], { stdin: 'pipe', stdout: 'pipe', stderr: 'pipe' });
    const out = readAll(proc.stdout);
    const err = readAll(proc.stderr);
    const writer = proc.stdin.getWriter();
    if (stdin) await writer.write(stdin);
    if (after) await after(proc, writer);
    else await writer.close();
    const status = await proc.wait();
    return { code: status.exit_status, signal: status.term_signal, out: await out, err: new TextDecoder().decode(await err) };
};

const cases = [];
const add = (name, fn) => cases.push([name, fn]);

add('round trip compresses English', async () => {
    const compressed = await zipper.compress(TEXT);
    assert.ok(compressed.length < TEXT.length);
    same(await zipper.decompress(compressed), TEXT);
});

add('round trip with window overlap', async () => {
    const data = concatBytes(TEXT, TEXT, TEXT, TEXT); // longer than the 128 token window
    const compressed = await zipper.compress(data, 32);
    same(await zipper.decompress(compressed, 32), data);
});

for (const [name, data] of EDGE_INPUTS) {
    add(`round trip edge input [${name}]`, async () => {
        same(await zipper.decompress(await zipper.compress(data)), data);
    });
}

// Additions: what the port has to get right that the Python tests cannot see.

add('compression is deterministic across a fresh model instance', async () => {
    const other = new LlamaZip(MODEL, { nCtx: 128, nGpuLayers: 0 });
    const data = concatBytes(TEXT, TEXT);
    same(await other.compress(data, 16), await zipper.compress(data, 16));
    other.close();
});

add('data compressed with one thread count is decoded with the same count only', async () => {
    const one = new LlamaZip(MODEL, { nCtx: 128, nGpuLayers: 0, nThreads: 1 });
    const compressed = await one.compress(TEXT);
    same(await one.decompress(compressed), TEXT);
    one.close();
});

add('decompress rejects garbage instead of looping', async () => {
    await assert.rejects(() => zipper.decompress(Uint8Array.of(0x55, 0xaa, 0x12, 0x34, 0x56, 0x78)), /ended before an end-of-stream token|not valid UTF-8/);
});

add('cli round trip, base64 argument', async () => {
    const compressed = await spawnCli(['-c', 'The quick brown fox jumps over the lazy dog.', '-f', 'base64', '--n-gpu-layers', '0']);
    assert.equal(compressed.code, 0, compressed.err);
    const packed = new TextDecoder().decode(compressed.out);
    const restored = await spawnCli(['-d', packed, '-f', 'base64', '--n-gpu-layers', '0']);
    assert.equal(restored.code, 0, restored.err);
    assert.equal(new TextDecoder().decode(restored.out), 'The quick brown fox jumps over the lazy dog.');
});

add('cli round trip, binary through stdin and stdout', async () => {
    const data = concatBytes(TEXT, Uint8Array.from([0xff, 0xfe, 0x80, 0x00, 0x01]), bytesOf(' café'));
    const compressed = await spawnCli(['-c', '--n-gpu-layers', '0'], { stdin: data });
    assert.equal(compressed.code, 0, compressed.err);
    assert.ok(compressed.err.includes('Compressing'));
    const restored = await spawnCli(['-d', '--n-gpu-layers', '0'], { stdin: compressed.out });
    assert.equal(restored.code, 0, restored.err);
    same(restored.out, data);
});

add('cli compression is interrupted by SIGINT and the output decodes to a prefix', async () => {
    const data = bytesOf('The quick brown fox jumps over the lazy dog. '.repeat(120));
    const result = await spawnCli(['-c', '--n-gpu-layers', '0', '-w', '32', '--n-ctx', '256'], {
        stdin: data,
        after: async (proc, writer) => {
            await writer.close();
            await new Promise(resolve => setTimeout(resolve, 2500));
            proc.kill('SIGINT');
        },
    });
    assert.equal(result.code, 0, result.err);
    assert.ok(result.out.length > 0);
    // Decode with the settings the child used: n-ctx 256 and overlap 32.
    const same256 = new LlamaZip(MODEL, { nCtx: 256, nGpuLayers: 0 });
    const prefix = await same256.decompress(result.out, 32);
    same256.close();
    assert.ok(prefix.length > 0 && prefix.length < data.length, `decoded ${prefix.length} of ${data.length} bytes`);
    same(prefix, data.slice(0, prefix.length));
});

add('cli interactive session through a pipe', async () => {
    const lines = 'Hi there!\n';
    const first = await spawnCli(['-i', '--n-gpu-layers', '0'], { stdin: bytesOf(lines) });
    assert.equal(first.code, 0, first.err);
    const packed = new TextDecoder().decode(first.out);
    assert.ok(packed.length > 0);
    assert.ok(first.err.includes('≥≥≥ '));
    const second = await spawnCli(['-i', '--n-gpu-layers', '0'], { stdin: bytesOf(`${packed}\nBye now.\n`) });
    assert.equal(second.code, 0, second.err);
    assert.ok(new TextDecoder().decode(second.out).startsWith('Hi there!'));
});

for (const [name, fn] of cases) {
    if (reason) skip(name, reason);
    else test(name, fn);
}

await run();
zipper?.close();
