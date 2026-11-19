// Command line tool, a port of main() in llama_zip.py.
//
//   tjs run js/cli.js [options] model_path (-c [STRING ...] | -d [COMPRESSED ...] | -i)
//
// Same flags and behaviour as the Python tool: -c, -d, -i, -f binary|base64, -w,
// --n-ctx, --n-gpu-layers, --use-mlock. --n-threads is new (see js/llamazip.js, both sides
// need the same value). The loading message, progress bar and prompts go to stderr, and
// compressed data and decompressed text go to stdout.
//
// SIGINT. During compression it stops early and finishes the stream with EOS, so the
// output decodes to the part that was compressed. During decompression and at the
// interactive prompt it throws KeyboardInterrupt like Python does. A signal that arrives while the
// model is loading or while a single llama_decode call runs is handled at the next
// event loop turn, which is at most about 50 ms later during compress and decompress.
import { LlamaZip, DEFAULT_N_THREADS } from './llamazip.js';
import { defaultIo } from './io.js';

export const PROG = 'cli.js';

export const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
export const BASE64_EQ = BASE64 + '=';

const BASE64_CODES = new Map([...BASE64].map((char, index) => [char.charCodeAt(0), index]));

export class KeyboardInterrupt extends Error {
    constructor() {
        super('KeyboardInterrupt');
        this.name = 'KeyboardInterrupt';
    }
}

// Raised for bad command lines. main() prints the usage and the message, and returns 2.
export class ArgumentError extends Error {}

export const b64encode = bytes => {
    let output = '';
    for (let i = 0; i < bytes.length; i += 3) {
        const chunk = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
        output += BASE64[(chunk >> 18) & 63] + BASE64[(chunk >> 12) & 63];
        output += i + 1 < bytes.length ? BASE64[(chunk >> 6) & 63] : '=';
        output += i + 2 < bytes.length ? BASE64[chunk & 63] : '=';
    }
    return output;
};

// Drops every byte outside the base64 alphabet (including "=") and pads with "A", which is
// zero bits, so the result may end in up to two extra zero bytes. The decoder reads zero
// bits past the end of the data anyway.
export const robustB64decode = inputBytes => {
    const filtered = Array.from(inputBytes).filter(byte => BASE64_CODES.has(byte));
    while (filtered.length % 4 !== 0) filtered.push(0x41);
    const output = new Uint8Array(filtered.length / 4 * 3);
    for (let i = 0, o = 0; i < filtered.length; i += 4, o += 3) {
        const chunk = (BASE64_CODES.get(filtered[i]) << 18) | (BASE64_CODES.get(filtered[i + 1]) << 12)
            | (BASE64_CODES.get(filtered[i + 2]) << 6) | BASE64_CODES.get(filtered[i + 3]);
        output[o] = chunk >> 16;
        output[o + 1] = (chunk >> 8) & 255;
        output[o + 2] = chunk & 255;
    }
    return output;
};

// What Python's int() and float() accept, so -w and --n-ctx behave the same: surrounding
// whitespace, a sign, and single underscores between digits.
const DIGITS = '\\d+(?:_\\d+)*';
const INT_RE = new RegExp(`^[+-]?${DIGITS}$`);
const FLOAT_RE = new RegExp(`^[+-]?(?:${DIGITS}(?:\\.(?:${DIGITS})?)?|\\.${DIGITS})(?:[eE][+-]?${DIGITS})?$`);
const SPECIAL_FLOAT_RE = /^([+-]?)(inf|infinity|nan)$/i;

export const pyInt = text => {
    const trimmed = text.trim();
    return INT_RE.test(trimmed) ? Number(trimmed.replaceAll('_', '')) : null;
};

export const pyFloat = text => {
    const trimmed = text.trim();
    const special = SPECIAL_FLOAT_RE.exec(trimmed);
    if (special) return special[2].toLowerCase() === 'nan' ? NaN : special[1] === '-' ? -Infinity : Infinity;
    return FLOAT_RE.test(trimmed) ? Number(trimmed.replaceAll('_', '')) : null;
};

// Window overlap in tokens from the -w argument, as main() computes it. Throws ArgumentError.
export const parseWindowOverlap = (argument, nCtx) => {
    const notInteger = "window overlap must be an integer (number of tokens) or a percentage (of the model's context length)";
    if (argument.endsWith('%')) {
        const percent = pyFloat(argument.slice(0, -1));
        if (percent === null) throw new ArgumentError(notInteger);
        if (!(percent >= 0 && percent <= 100)) throw new ArgumentError('window overlap must be in the range [0%, 100%]');
        return Math.trunc(percent / 100 * (nCtx - 1));
    }
    let overlap = pyInt(argument);
    if (overlap === null) throw new ArgumentError(notInteger);
    if (overlap < 0) overlap += nCtx;
    if (!(overlap >= 0 && overlap < nCtx)) throw new ArgumentError(`window overlap must be in the range [${-nCtx}, ${nCtx - 1}]`);
    return overlap;
};

const USAGE = `usage: ${PROG} [-h] [-f {binary,base64}] [-w OVERLAP] [--n-ctx N_CTX]
       ${' '.repeat(PROG.length)} [--n-gpu-layers N_GPU_LAYERS] [--n-threads N_THREADS] [--use-mlock]
       ${' '.repeat(PROG.length)} (-c [STRING ...] | -d [COMPRESSED ...] | -i)
       ${' '.repeat(PROG.length)} model_path
`;

const HELP = `${USAGE}
LLM-powered lossless compression tool

positional arguments:
  model_path            path to model file

options:
  -h, --help            show this help message and exit
  -f, --compressed-format {binary,base64}
                        format of compressed data (default: binary, except for interactive mode, which only supports base64)
  -w, --window-overlap OVERLAP
                        how much model context (as number of tokens or percentage of model context length) to maintain after filling the window. higher values increase compression ratio but decrease speed. must use same value for compression and decompression (default: 0%)
  --n-ctx N_CTX         model context length (default: 0, which uses maximum supported by the model)
  --n-gpu-layers N_GPU_LAYERS
                        number of model layers to offload to GPU (default: -1, which offloads all layers)
  --n-threads N_THREADS
                        number of CPU threads. must use same value for compression and decompression (default: ${DEFAULT_N_THREADS})
  --use-mlock           use mlock to keep model in RAM (disabled by default)
  -c, --compress [STRING ...]
                        compress argument string (or stdin if no argument is provided)
  -d, --decompress [COMPRESSED ...]
                        decompress argument string (or stdin if no argument is provided)
  -i, --interactive     show a prompt for interactive compression and decompression
`;

const intType = text => {
    const value = pyInt(text);
    if (value === null) throw new ArgumentError(`invalid int value: '${text}'`);
    return value;
};

const OPTIONS = [
    { flags: ['-h', '--help'], dest: 'help', kind: 'flag' },
    { flags: ['-f', '--compressed-format'], dest: 'compressedFormat', kind: 'value', choices: ['binary', 'base64'] },
    { flags: ['-w', '--window-overlap'], dest: 'overlap', kind: 'value' },
    { flags: ['--n-ctx'], dest: 'nCtx', kind: 'value', type: intType },
    { flags: ['--n-gpu-layers'], dest: 'nGpuLayers', kind: 'value', type: intType },
    { flags: ['--n-threads'], dest: 'nThreads', kind: 'value', type: intType },
    { flags: ['--use-mlock'], dest: 'useMlock', kind: 'flag' },
    { flags: ['-c', '--compress'], dest: 'string', kind: 'list', mode: true },
    { flags: ['-d', '--decompress'], dest: 'compressed', kind: 'list', mode: true },
    { flags: ['-i', '--interactive'], dest: 'interactive', kind: 'flag', mode: true },
];

const optionName = option => option.flags.join('/');

// Negative numbers are values, not options, because no option looks like one.
const NEGATIVE_NUMBER_RE = /^-\d+$|^-\d*\.\d+$/;

// Like argparse, text with a space is a value unless it is a known long option with "=value", such as "--window-overlap= 7 ".
const looksLikeOption = arg => {
    if (arg.length < 2 || arg[0] !== '-' || NEGATIVE_NUMBER_RE.test(arg)) return false;
    if (!arg.includes(' ')) return true;
    return arg.startsWith('--') && arg.includes('=') && OPTIONS.some(option => option.flags.includes(arg.split('=', 1)[0]));
};

const findOption = arg => {
    if (arg.startsWith('--')) {
        const [name] = arg.split('=', 1);
        const exact = OPTIONS.find(option => option.flags.includes(name));
        if (exact) return exact;
        const candidates = OPTIONS.filter(option => option.flags.some(flag => flag.startsWith('--') && flag.startsWith(name)));
        if (candidates.length > 1) {
            throw new ArgumentError(`ambiguous option: ${name} could match ${candidates.flatMap(option => option.flags.filter(flag => flag.startsWith(name))).join(', ')}`);
        }
        return candidates[0] ?? null;
    }
    return OPTIONS.find(option => option.flags.includes(arg.slice(0, 2))) ?? null;
};

// Parses like the argparse parser in llama_zip.py. Returns the options, or { help: true }.
// Not supported: clustered short flags such as -ci.
export const parseArgs = argv => {
    const args = {
        modelPath: undefined,
        compressedFormat: null,
        overlap: '0%',
        nCtx: 0,
        nGpuLayers: -1,
        nThreads: DEFAULT_N_THREADS,
        useMlock: false,
        string: null,
        compressed: null,
        interactive: false,
    };
    const positionals = [];
    const unrecognized = [];
    let mode = null;
    let onlyPositionals = false;
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (onlyPositionals || !looksLikeOption(arg)) {
            positionals.push(arg);
            continue;
        }
        if (arg === '--') {
            onlyPositionals = true;
            continue;
        }
        const option = findOption(arg);
        if (!option) {
            unrecognized.push(arg);
            continue;
        }
        if (option.dest === 'help') return { help: true };
        let explicit = null;
        if (arg.startsWith('--')) {
            if (arg.includes('=')) explicit = arg.slice(arg.indexOf('=') + 1);
        }
        else if (arg.length > 2) {
            explicit = arg.slice(arg[2] === '=' ? 3 : 2);
        }
        if (option.mode) {
            if (mode && mode !== option) throw new ArgumentError(`argument ${optionName(option)}: not allowed with argument ${optionName(mode)}`);
            mode = option;
        }
        if (option.kind === 'flag') {
            if (explicit !== null) throw new ArgumentError(`argument ${optionName(option)}: ignored explicit argument '${explicit}'`);
            args[option.dest] = true;
        }
        else if (option.kind === 'value') {
            let value = explicit;
            if (value === null) {
                if (i + 1 >= argv.length || (looksLikeOption(argv[i + 1]))) throw new ArgumentError(`argument ${optionName(option)}: expected one argument`);
                value = argv[++i];
            }
            if (option.choices && !option.choices.includes(value)) {
                throw new ArgumentError(`argument ${optionName(option)}: invalid choice: '${value}' (choose from ${option.choices.map(choice => `'${choice}'`).join(', ')})`);
            }
            try {
                args[option.dest] = option.type ? option.type(value) : value;
            }
            catch (error) {
                if (error instanceof ArgumentError) throw new ArgumentError(`argument ${optionName(option)}: ${error.message}`);
                throw error;
            }
        }
        else {
            const values = explicit !== null ? [explicit] : [];
            while (i + 1 < argv.length && !looksLikeOption(argv[i + 1])) values.push(argv[++i]);
            args[option.dest] = values;
        }
    }
    if (positionals.length > 0) args.modelPath = positionals.shift();
    if (args.modelPath === undefined) throw new ArgumentError('the following arguments are required: model_path');
    if (!mode) throw new ArgumentError('one of the arguments -c/--compress -d/--decompress -i/--interactive is required');
    unrecognized.push(...positionals);
    if (unrecognized.length > 0) throw new ArgumentError(`unrecognized arguments: ${unrecognized.join(' ')}`);
    return args;
};

const concat = chunks => {
    const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
        output.set(chunk, offset);
        offset += chunk.length;
    }
    return output;
};

export const readAllStdin = async () => {
    const reader = tjs.stdin.getReader();
    const chunks = [];
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    reader.releaseLock();
    return concat(chunks);
};

// Reads lines (as bytes) from stdin without losing data when a read is abandoned by a signal.
class LineReader {
    #reader = tjs.stdin.getReader();
    #buffer = new Uint8Array(0);
    #pending = null;
    #eof = false;

    // Resolves to the line without its newline, or null at the end of input. Rejects with the
    // signal's reason when it aborts.
    async readLine(signal) {
        for (;;) {
            const newline = this.#buffer.indexOf(10);
            if (newline >= 0 || this.#eof) {
                if (newline < 0 && this.#buffer.length === 0) return null;
                const end = newline < 0 ? this.#buffer.length : newline;
                let line = this.#buffer.slice(0, end);
                this.#buffer = this.#buffer.slice(end + 1);
                if (line[line.length - 1] === 13) line = line.slice(0, -1);
                return line;
            }
            this.#pending ??= this.#reader.read();
            const aborted = new Promise((_, reject) => {
                if (signal.aborted) reject(signal.reason);
                signal.addEventListener('abort', () => reject(signal.reason), { once: true });
            });
            const result = await Promise.race([this.#pending, aborted]);
            this.#pending = null;
            if (result.done) this.#eof = true;
            else this.#buffer = concat([this.#buffer, result.value]);
        }
    }
}

let lineReader = null;

// Writes the prompt to stderr like Python's input() does.
const defaultReadLine = async (prompt, { signal, io }) => {
    io.err.write(prompt);
    await io.err.flush();
    lineReader ??= new LineReader();
    return lineReader.readLine(signal);
};

export const defaultOnSigint = handler => {
    tjs.addSignalListener('SIGINT', handler);
    return () => tjs.removeSignalListener('SIGINT', handler);
};

const strictDecoder = () => new TextDecoder('utf-8', { fatal: true });

// Runs main() with injectable parts for tests. Returns the exit code.
//
// deps (all optional):
//   io           { out, err } sinks with write() and flush()
//   createZip    async (modelPath, options) => LlamaZip, default LlamaZip.load
//   readStdin    async () => Uint8Array
//   readLine     async (prompt, { signal, io }) => Uint8Array | string | null (null is end of input)
//   onSigint     handler => unsubscribe
export const main = async (argv, deps = {}) => {
    const io = deps.io ?? defaultIo();
    const createZip = deps.createZip ?? ((path, options) => LlamaZip.load(path, options));
    const readStdin = deps.readStdin ?? readAllStdin;
    const readLine = deps.readLine ?? defaultReadLine;
    const onSigint = deps.onSigint ?? defaultOnSigint;
    const fail = error => {
        io.err.write(`${USAGE}${PROG}: error: ${error.message}\n`);
        return 2;
    };
    const finish = async code => {
        await io.out.flush();
        await io.err.flush();
        return code;
    };

    let args;
    try {
        args = parseArgs(argv);
        if (args.help) {
            io.out.write(HELP);
            return await finish(0);
        }
        if (args.compressedFormat === null) args.compressedFormat = args.interactive ? 'base64' : 'binary';
        else if (args.interactive && args.compressedFormat !== 'base64') {
            throw new ArgumentError('interactive mode only supports base64 compressed data');
        }
    }
    catch (error) {
        if (!(error instanceof ArgumentError)) throw error;
        return await finish(fail(error));
    }

    // Runs fn with an AbortSignal that SIGINT aborts with a KeyboardInterrupt.
    const interruptible = async fn => {
        const controller = new AbortController();
        const unsubscribe = onSigint(() => controller.abort(new KeyboardInterrupt()));
        try {
            return await fn(controller.signal);
        }
        finally {
            unsubscribe();
        }
    };

    const zip = await createZip(args.modelPath, {
        nCtx: args.nCtx,
        nGpuLayers: args.nGpuLayers,
        useMlock: args.useMlock,
        nThreads: args.nThreads,
        verbose: true,
        io,
    });
    try {
        const windowOverlap = parseWindowOverlap(args.overlap, zip.model.nCtx);
        const compress = bytes => interruptible(signal => zip.compress(bytes, windowOverlap, { signal }));
        const decompress = bytes => interruptible(signal => zip.decompress(bytes, windowOverlap, { signal }));

        if (args.string !== null) {
            const uncompressed = args.string.length > 0 ? new TextEncoder().encode(args.string.join(' ')) : await readStdin();
            const compressed = await compress(uncompressed);
            io.out.write(args.compressedFormat === 'base64' ? b64encode(compressed) : compressed);
        }
        else if (args.compressed !== null) {
            let compressed = args.compressed.length > 0 ? new TextEncoder().encode(args.compressed[0]) : await readStdin();
            if (args.compressedFormat === 'base64') compressed = robustB64decode(compressed);
            await decompress(compressed);
        }
        else if (args.interactive) {
            for (;;) {
                const line = await interruptible(signal => readLine('≥≥≥ ', { signal, io }));
                if (line === null) {
                    io.err.write('\n');
                    break;
                }
                const inputBytes = typeof line === 'string' ? new TextEncoder().encode(line) : line;
                try {
                    strictDecoder().decode(inputBytes);
                }
                catch {
                    io.err.write('error: interactive mode only supports UTF-8 input\n');
                    continue;
                }
                if (inputBytes.length > 0 && inputBytes.every(byte => BASE64_EQ.includes(String.fromCharCode(byte)))) {
                    try {
                        await decompress(robustB64decode(inputBytes));
                    }
                    catch (error) {
                        if (!(error instanceof KeyboardInterrupt)) throw error;
                    }
                }
                else {
                    const compressed = await compress(inputBytes);
                    io.out.write(b64encode(compressed));
                }
                io.err.write('\n\n');
            }
        }
    }
    catch (error) {
        if (error instanceof ArgumentError) return await finish(fail(error));
        if (!(error instanceof KeyboardInterrupt)) throw error;
        io.err.write('\n');
    }
    finally {
        zip.close?.();
    }
    return await finish(0);
};

if (import.meta.main) {
    const offset = tjs.args[1] === 'run' ? 3 : 2;
    let code;
    try {
        code = await main(tjs.args.slice(offset));
    }
    catch (error) {
        console.error(`error: ${error?.message ?? error}`);
        code = 1;
    }
    tjs.exit(code);
}
