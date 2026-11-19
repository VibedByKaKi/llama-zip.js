// Port of tests/python/test_cli.py. Run with `tjs run js/test/cli.test.js`.
//
// main() takes its streams, stdin, line input, SIGINT hook and model loader as arguments,
// which plays the part of the monkeypatching in the Python tests.
import { KeyboardInterrupt, b64encode, defaultOnSigint, main, parseWindowOverlap, pyFloat, pyInt, robustB64decode } from '../cli.js';
import { LlamaZip } from '../llamazip.js';
import { FakeModel, captureIo, createModel, fakeTest as test } from './fake_llama.js';
import { assert, run, skip } from './run.js';
import { bytesOf, fromHex, loadVectors, toHex } from './vectors.js';

const WINDOW = await loadVectors('window_overlap.json');

// Runs main() against the fake model like the `run` fixture. `patch` edits the LlamaZip after it loads,
// `inputs` are the lines the interactive prompt returns (an Error is thrown instead).
const runCli = async (argv, { stdin = new Uint8Array(0), inputs = null, patch = null, sigint = null, withModel = true } = {}) => {
    const io = captureIo();
    const queue = inputs ? [...inputs] : [];
    const deps = {
        io,
        readStdin: async () => stdin,
        createZip: async (path, options) => {
            const zip = await LlamaZip.load(path, { ...options, createModel });
            patch?.(zip);
            return zip;
        },
        readLine: async () => {
            io.err.write('[prompt]');
            const item = queue.shift();
            if (item instanceof Error) throw item;
            return item;
        },
        onSigint: sigint ?? (() => () => {}),
    };
    const code = await main([...(withModel ? ['fake.gguf'] : []), ...argv], deps);
    return { code, out: io.outBytes(), err: io.errText() };
};

const reference = async (data, overlap = 0, nCtx = 64) => LlamaZip.prototype.compress.call(new LlamaZip('fake.gguf', { nCtx, createModel }), data, overlap);
const b64 = async (...args) => bytesOf(b64encode(await reference(...args)));
const same = (actual, expected, message) => assert.equal(toHex(actual), toHex(expected), message);
const text = value => bytesOf(value);
const cat = (...parts) => Uint8Array.from(parts.flatMap(part => Array.from(part)));
const interrupt = () => new KeyboardInterrupt();

// TestArgumentErrors

test('ArgumentErrors mode is required', async () => {
    const { code, err } = await runCli([]);
    assert.equal(code, 2);
    assert.ok(err.includes('one of the arguments -c/--compress -d/--decompress -i/--interactive is required'));
});

test('ArgumentErrors modes are mutually exclusive', async () => {
    const { code, err } = await runCli(['-c', 'x', '-d', 'y']);
    assert.equal(code, 2);
    assert.ok(err.includes('not allowed with argument'));
});

test('ArgumentErrors compress and interactive are exclusive', async () => {
    assert.equal((await runCli(['-c', '-i'])).code, 2);
});

test('ArgumentErrors unknown format is rejected', async () => {
    const { code, err } = await runCli(['-c', 'x', '-f', 'hex']);
    assert.equal(code, 2);
    assert.ok(err.includes('invalid choice'));
});

test('ArgumentErrors interactive rejects binary', async () => {
    const { code, err } = await runCli(['-i', '-f', 'binary']);
    assert.equal(code, 2);
    assert.ok(err.includes('interactive mode only supports base64 compressed data'));
});

test('ArgumentErrors missing model path', async () => {
    const { code } = await runCli(['-c', 'x'], { withModel: false });
    assert.equal(code, 2);
});

test('ArgumentErrors n-ctx must be an integer', async () => {
    const { code, err } = await runCli(['-c', 'x', '--n-ctx', 'big']);
    assert.equal(code, 2);
    assert.ok(err.includes('invalid int value'));
});

test('ArgumentErrors help exits cleanly', async () => {
    const { code, out } = await runCli(['-h']);
    assert.equal(code, 0);
    const help = new TextDecoder().decode(out);
    assert.ok(help.includes('LLM-powered lossless compression tool'));
    assert.ok(help.includes('default: 0%'));
});

test('ArgumentErrors arguments error before the model loads', async () => {
    await runCli(['-i', '-f', 'binary']);
    assert.equal(FakeModel.instances.length, 0);
});

// Additions for the hand written parser (argparse is not available).

test('ArgumentErrors unrecognized arguments are listed', async () => {
    const { code, err } = await runCli(['-c', 'x', '--bogus', 'extra']);
    assert.equal(code, 2);
    assert.ok(err.includes('unrecognized arguments: --bogus'));
});

test('ArgumentErrors an option without its value is reported', async () => {
    const { code, err } = await runCli(['-c', 'x', '-f']);
    assert.equal(code, 2);
    assert.ok(err.includes('argument -f/--compressed-format: expected one argument'));
});

test('ArgumentErrors the error output has a usage line', async () => {
    const { err } = await runCli(['-c', 'x', '--n-ctx', 'big']);
    assert.ok(err.startsWith('usage: '));
    assert.ok(err.includes('cli.js: error: argument --n-ctx: invalid int value'));
});

test('ArgumentErrors long options can be abbreviated like in argparse', async () => {
    const { seen, patch } = spyOnCompress();
    const { code } = await runCli(['--compress', 'x', '--window-ov', '3'], { patch });
    assert.equal(code, 0);
    assert.deepEqual(seen, [3]);
    // "--compr" matches both --compress and --compressed-format.
    const ambiguous = await runCli(['--compr', 'x']);
    assert.equal(ambiguous.code, 2);
    assert.ok(ambiguous.err.includes('ambiguous option'));
});

// TestModelOptions

test('ModelOptions defaults are forwarded', async () => {
    await runCli(['-c', 'x']);
    assert.equal(FakeModel.instances.length, 1);
    const [model] = FakeModel.instances;
    assert.equal(model.path, 'fake.gguf');
    assert.deepEqual([model.nCtx, model.nGpuLayers, model.useMlock], [64, -1, false]);
    assert.equal(model.nThreads, 4);
});

test('ModelOptions options are forwarded', async () => {
    await runCli(['-c', 'x', '--n-ctx', '48', '--n-gpu-layers', '7', '--use-mlock', '--n-threads', '2']);
    const [model] = FakeModel.instances;
    assert.deepEqual([model.nCtx, model.nGpuLayers, model.useMlock, model.nThreads], [48, 7, true, 2]);
});

test('ModelOptions loading message goes to stderr', async () => {
    const { out, err } = await runCli(['-c', 'x']);
    assert.ok(err.includes('Loading model...'));
    assert.ok(!new TextDecoder().decode(out).includes('Loading'));
});

// TestCompress

test('Compress binary output from arguments', async () => {
    const { code, out } = await runCli(['-c', 'the', 'thing']);
    assert.equal(code, 0);
    same(out, await reference(text('the thing')));
});

test('Compress arguments are joined with spaces', async () => {
    same((await runCli(['-c', 'a', 'b', 'c'])).out, await reference(text('a b c')));
});

test('Compress non ASCII argument', async () => {
    same((await runCli(['-c', 'café €'])).out, await reference(text('café €')));
});

test('Compress reads stdin without arguments', async () => {
    const data = Uint8Array.from([...text('from stdin '), 0xff, 0xfe, 0x0a]);
    same((await runCli(['-c'], { stdin: data })).out, await reference(data));
});

test('Compress empty stdin', async () => {
    same((await runCli(['-c'], { stdin: new Uint8Array(0) })).out, await reference(new Uint8Array(0)));
});

test('Compress base64 output', async () => {
    const { out } = await runCli(['-c', 'the thing', '-f', 'base64']);
    same(out, await b64(text('the thing')));
});

test('Compress long option names', async () => {
    const { out } = await runCli(['--compress', 'x', '--compressed-format', 'base64']);
    same(out, await b64(text('x')));
});

test('Compress window overlap changes the output', async () => {
    const data = 'the thing, the king; '.repeat(8);
    const a = (await runCli(['-c', data, '--n-ctx', '16', '-w', '0'])).out;
    const b = (await runCli(['-c', data, '--n-ctx', '16', '-w', '5'])).out;
    same(a, await reference(text(data), 0, 16));
    same(b, await reference(text(data), 5, 16));
    assert.notEqual(toHex(a), toHex(b));
});

test('Compress progress bar goes to stderr only', async () => {
    const { out } = await runCli(['-c', 'the thing']);
    assert.ok(!new TextDecoder().decode(out).includes('Compressing'));
});

test('Compress keyboard interrupt is swallowed', async () => {
    const { code, out, err } = await runCli(['-c', 'x'], {
        patch: zip => { zip.compress = async () => { throw interrupt(); }; },
    });
    assert.equal(code, 0);
    assert.equal(out.length, 0);
    assert.ok(err.endsWith('\n'));
});

// JS only: the SIGINT hook, which stands in for signal.signal() in Python's compress().

test('Compress SIGINT stops early, finishes with EOS and the output still decodes', async () => {
    let handler = null;
    let unsubscribed = 0;
    const sigint = fn => {
        handler = fn;
        return () => { unsubscribed++; };
    };
    let count = 0;
    FakeModel.onLogits = () => {
        if (++count === 4) handler();
    };
    const { code, out } = await runCli(['-c', 'abcdefghij'], { sigint });
    FakeModel.onLogits = null;
    assert.equal(code, 0);
    assert.equal(unsubscribed, 1);
    const z = new LlamaZip('fake.gguf', { nCtx: 64, createModel });
    same(await z.decompress(out), text('abc'));
});

// TestDecompress

test('Decompress base64 argument', async () => {
    const packed = new TextDecoder().decode(await b64(text('the thing')));
    const { code, out } = await runCli(['-d', packed, '-f', 'base64']);
    assert.equal(code, 0);
    same(out, text('the thing'));
});

test('Decompress base64 argument is cleaned up', async () => {
    const packed = new TextDecoder().decode(await b64(text('the thing')));
    const noisy = packed.slice(0, 4) + ' \n' + packed.slice(4) + '\n';
    same((await runCli(['-d', noisy, '-f', 'base64'])).out, text('the thing'));
});

test('Decompress binary from stdin', async () => {
    const data = Uint8Array.from([...text('binary '), 0xff, 0xfe, 0x80, ...text(' data')]);
    same((await runCli(['-d'], { stdin: await reference(data) })).out, data);
});

test('Decompress base64 from stdin', async () => {
    const packed = cat(await b64(text('x y z')), text('\n'));
    same((await runCli(['-d', '-f', 'base64'], { stdin: packed })).out, text('x y z'));
});

test('Decompress only the first argument is used', async () => {
    const packed = new TextDecoder().decode(await b64(text('first')));
    const other = new TextDecoder().decode(await b64(text('second')));
    same((await runCli(['-d', packed, other, '-f', 'base64'])).out, text('first'));
});

test('Decompress window overlap round trip', async () => {
    const data = text('the thing, the king; '.repeat(8));
    const packed = new TextDecoder().decode(await b64(data, 5, 16));
    same((await runCli(['-d', packed, '-f', 'base64', '--n-ctx', '16', '-w', '5'])).out, data);
});

test('Decompress with percent overlap', async () => {
    const data = text('the thing, the king; '.repeat(8));
    const overlap = Math.trunc(0.5 * (16 - 1));
    const packed = new TextDecoder().decode(await b64(data, overlap, 16));
    same((await runCli(['-d', packed, '-f', 'base64', '--n-ctx', '16', '-w', '50%'])).out, data);
});

test('Decompress keyboard interrupt is swallowed', async () => {
    const { code, err } = await runCli(['-d', 'AAAA', '-f', 'base64'], {
        patch: zip => { zip.decompress = async () => { throw interrupt(); }; },
    });
    assert.equal(code, 0);
    assert.ok(err.endsWith('\n'));
});

test('Decompress SIGINT aborts decompression and prints a newline', async () => {
    const data = text('abcdefghij');
    let handler = null;
    const sigint = fn => {
        handler = fn;
        return () => {};
    };
    const compressed = b64encode(await reference(data));
    let count = 0;
    FakeModel.onLogits = () => {
        if (++count === 4) handler();
    };
    const { code, out, err } = await runCli(['-d', compressed, '-f', 'base64'], { sigint });
    FakeModel.onLogits = null;
    assert.equal(code, 0);
    same(out, text('abc'));
    assert.ok(err.endsWith('\n'));
});

test('Decompress data without an end of stream token is an error, not a hang', async () => {
    let failure = null;
    try {
        await runCli(['-d', 'AAAA', '-f', 'base64']);
    }
    catch (error) {
        failure = error;
    }
    assert.ok(failure && /ended before an end-of-stream token/.test(failure.message), String(failure));
});

// TestWindowOverlapParsing

const spyOnCompress = () => {
    const seen = [];
    return { seen, patch: zip => { zip.compress = async (data, overlap = 0) => { seen.push(overlap); return new Uint8Array(0); }; } };
};

for (const c of WINDOW.cases) {
    test(`WindowOverlapParsing vectors [ctx${c.n_ctx}-${JSON.stringify(c.arg)}]`, async () => {
        const { seen, patch } = spyOnCompress();
        const { code, err } = await runCli(['-c', 'x', '--n-ctx', String(c.n_ctx), `--window-overlap=${c.arg}`], { patch });
        if ('overlap' in c) {
            assert.equal(code, 0, err);
            assert.deepEqual(seen, [c.overlap]);
        }
        else {
            assert.equal(code, 2);
            assert.deepEqual(seen, []);
            const template = WINDOW.messages[c.error];
            assert.ok(err.includes(template.replace('{n_ctx}', c.n_ctx).replace('{n_ctx_minus_1}', c.n_ctx - 1)));
        }
    });
}

test('WindowOverlapParsing default overlap is zero', async () => {
    const { seen, patch } = spyOnCompress();
    await runCli(['-c', 'x'], { patch });
    assert.deepEqual(seen, [0]);
});

test('WindowOverlapParsing negative number as a separate argument', async () => {
    const { seen, patch } = spyOnCompress();
    await runCli(['-c', 'x', '--n-ctx', '64', '-w', '-1'], { patch });
    assert.deepEqual(seen, [63]);
});

test('WindowOverlapParsing percentage uses n_ctx minus one', async () => {
    const { seen, patch } = spyOnCompress();
    await runCli(['-c', 'x', '--n-ctx', '11', '-w', '100%'], { patch });
    assert.deepEqual(seen, [10]);
});

test('WindowOverlapParsing message is a parser error', async () => {
    const { err } = await runCli(['-c', 'x', '-w', 'oops']);
    assert.ok(err.includes('usage:'));
    assert.ok(err.includes(': error: window overlap must be an integer'));
});

// JS only: the Python number parsers behind those vectors.

test('pyInt and pyFloat accept what Python accepts', () => {
    assert.equal(pyInt(' 7 '), 7);
    assert.equal(pyInt('+3'), 3);
    assert.equal(pyInt('-12'), -12);
    assert.equal(pyInt('1_0'), 10);
    for (const bad of ['', '1__0', '_1', '1_', '1.5', '1e2', '0x10', 'abc', '+', '--1']) assert.equal(pyInt(bad), null, bad);
    assert.equal(pyFloat('5 '), 5);
    assert.equal(pyFloat('1_0.5'), 10.5);
    assert.equal(pyFloat('.5'), 0.5);
    assert.equal(pyFloat('1.'), 1);
    assert.equal(pyFloat('1e2'), 100);
    assert.equal(pyFloat('+inf'), Infinity);
    assert.equal(pyFloat('-Infinity'), -Infinity);
    assert.ok(Number.isNaN(pyFloat('NaN')));
    for (const bad of ['', '.', 'e5', '1_', '1._5', 'ten', '1 2']) assert.equal(pyFloat(bad), null, bad);
});

test('parseWindowOverlap truncates percentages toward zero', () => {
    assert.equal(parseWindowOverlap('33.3%', 100), 32);
    assert.equal(parseWindowOverlap('0.5%', 1000), 4);
    assert.equal(parseWindowOverlap('-100', 100), 0);
});

// TestInteractive

const promptCount = err => err.split('[prompt]').length - 1;

test('Interactive compress then exit on ctrl-c', async () => {
    const { code, out, err } = await runCli(['-i'], { inputs: ['Hi there!', interrupt()] });
    assert.equal(code, 0);
    same(out, await b64(text('Hi there!')));
    assert.equal(promptCount(err), 2);
});

test('Interactive base64 input is decompressed', async () => {
    const packed = new TextDecoder().decode(await b64(text('Hi there!')));
    const { out } = await runCli(['-i'], { inputs: [packed, interrupt()] });
    same(out, text('Hi there!'));
});

test('Interactive compress decompress session', async () => {
    const packed = new TextDecoder().decode(await b64(text('Hi there!')));
    const { out, err } = await runCli(['-i'], { inputs: ['Hi there!', packed, 'Bye now.', interrupt()] });
    same(out, cat(await b64(text('Hi there!')), text('Hi there!'), await b64(text('Bye now.'))));
    assert.equal(promptCount(err), 4);
});

test('Interactive each turn ends with a blank line on stderr', async () => {
    const { err } = await runCli(['-i'], { inputs: ['Hi there!', interrupt()] });
    assert.ok(err.includes('\n\n[prompt]'));
});

test('Interactive empty line compresses the empty string', async () => {
    const { out } = await runCli(['-i'], { inputs: ['', interrupt()] });
    same(out, await b64(new Uint8Array(0)));
});

const spyOnDecompress = () => {
    const calls = [];
    return { calls, patch: zip => { zip.decompress = async data => { calls.push(Array.from(data)); return new Uint8Array(0); }; } };
};

test('Interactive plain word made of base64 characters is treated as base64', async () => {
    // "hello" only uses base64 characters, so interactive mode decodes it.
    const { calls, patch } = spyOnDecompress();
    await runCli(['-i'], { inputs: ['hello', interrupt()], patch });
    assert.deepEqual(calls, [Array.from(robustB64decode(text('hello')))]);
});

test('Interactive equals only input is routed to decompress', async () => {
    const { calls, patch } = spyOnDecompress();
    await runCli(['-i'], { inputs: ['===', interrupt()], patch });
    assert.deepEqual(calls, [[]]);
});

test('Interactive non base64 character makes it a compression request', async () => {
    const calls = [];
    const patch = zip => { zip.compress = async data => { calls.push(toHex(data)); return new Uint8Array(0); }; };
    await runCli(['-i'], { inputs: ['abc-def', 'AAAA!', interrupt()], patch });
    assert.deepEqual(calls, [toHex(text('abc-def')), toHex(text('AAAA!'))]);
});

test('Interactive unicode error is reported and the loop continues', async () => {
    const { out, err } = await runCli(['-i'], { inputs: [Uint8Array.of(0xff), 'Hi there!', interrupt()] });
    assert.ok(err.includes('error: interactive mode only supports UTF-8 input'));
    same(out, await b64(text('Hi there!')));
});

test('Interactive interrupting a decompression returns to the prompt', async () => {
    let calls = 0;
    const patch = zip => {
        zip.decompress = async () => {
            calls++;
            if (calls === 1) throw interrupt();
            return new Uint8Array(0);
        };
    };
    const { code, err } = await runCli(['-i'], { inputs: ['AAAA', 'BBBB', interrupt()], patch });
    assert.equal(code, 0);
    assert.equal(calls, 2);
    assert.equal(promptCount(err), 3);
});

test('Interactive window overlap applies in interactive mode', async () => {
    const { seen, patch } = spyOnCompress();
    await runCli(['-i', '--n-ctx', '20', '-w', '50%'], { inputs: ['Hi there!', interrupt()], patch });
    assert.deepEqual(seen, [9]);
});

test('Interactive explicit base64 format is accepted', async () => {
    const { code, out } = await runCli(['-i', '-f', 'base64'], { inputs: ['Hi there!', interrupt()] });
    assert.equal(code, 0);
    assert.ok(out.length > 0);
});

// JS only: end of input at the prompt (Python's input() raises EOFError) and the SIGINT subscription.

test('Interactive end of input leaves cleanly', async () => {
    const { code, out, err } = await runCli(['-i'], { inputs: ['Hi there!', null] });
    assert.equal(code, 0);
    same(out, await b64(text('Hi there!')));
    assert.equal(promptCount(err), 2);
});

test('Interactive carriage return line endings are not part of the input', async () => {
    // LineReader strips them, the prompt callback here returns what it would return.
    const { out } = await runCli(['-i'], { inputs: [text('Hi there!'), null] });
    same(out, await b64(text('Hi there!')));
});

test('defaultOnSigint subscribes to the real SIGINT and unsubscribes', async () => {
    let fired = 0;
    const unsubscribe = defaultOnSigint(() => { fired++; });
    tjs.kill(tjs.pid, 'SIGINT');
    // The event loop needs a turn, and a signal is only noticed after the timer's first 1 ms.
    for (let i = 0; i < 20 && fired === 0; i++) await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(fired, 1);
    unsubscribe();
});

skip('no mode selected does nothing', 'the parser rejects a missing mode, so this branch of Python main() does not exist in the port');

test('running as a script prints the help', async () => {
    const script = new URL('../cli.js', import.meta.url).pathname;
    const proc = tjs.spawn([tjs.exePath, 'run', script, '--help'], { stdout: 'pipe', stderr: 'pipe' });
    const reader = proc.stdout.getReader();
    const chunks = [];
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const status = await proc.wait();
    assert.equal(status.exit_status, 0);
    assert.ok(new TextDecoder().decode(cat(...chunks)).includes('LLM-powered lossless compression tool'));
});

test('running as a script reports a bad model path', async () => {
    const script = new URL('../cli.js', import.meta.url).pathname;
    const proc = tjs.spawn([tjs.exePath, 'run', script, '/nonexistent/model.gguf', '-c', 'x'], { stdout: 'pipe', stderr: 'pipe' });
    const reader = proc.stderr.getReader();
    const chunks = [];
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const status = await proc.wait();
    assert.equal(status.exit_status, 1);
    assert.ok(/error: /.test(new TextDecoder().decode(cat(...chunks))));
});

await run();
