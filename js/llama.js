// Minimal binding to the system libllama through tjs:ffi (txiki.js).
//
// Targets llama.cpp 1.0.0 (build 12359), the llama.h of that release. The struct
// layouts below are written by hand for that header. A struct passed by value
// with the wrong layout corrupts the call without any error, and the library
// cannot tell us its layout, so the binding asks llama_version() and accepts
// only the versions in KNOWN_VERSIONS. Older libraries have no llama_version()
// and are rejected. Another version is rejected too, unless the environment
// variable LLAMA_ZIP_ALLOW_UNTESTED=1 is set. The expected sizes and offsets are
// in js/test/llama.test.js, generated from the real header with a C compiler.
//
// API
//   const model = new LlamaModel({ path, nCtx = 0, nGpuLayers = 0,
//                                  useMlock = false, nBatch = 512,
//                                  nThreads = 0, verbose = false });
//     path        GGUF file.
//     nCtx        context length, 0 means the model's trained context (like llama-cpp-python).
//     nGpuLayers  layers to offload, 0 (default) keeps everything on the CPU.
//     nBatch      tokens per llama_decode call, eval() chunks to this size.
//     nThreads    0 keeps the libllama default.
//     verbose     true leaves llama.cpp logging on stderr, false (default) drops it.
//
//   model.nCtx                    actual context length in tokens (llama.cpp raises tiny values to 256)
//   model.nBatch                  effective batch size, nBatch clamped to nCtx
//   model.nVocab                  vocabulary size (length of logits())
//   model.nPast                   tokens currently in the context
//   model.llamaVersion            version reported by the library, e.g. '1.0.0'
//   model.llamaBuild              version and build number, e.g. '1.0.0 (build 12359)'
//                                 (only the version when the build number is unknown)
//   libraryInfo()                 { version, build, text } of the library, without opening a model
//   model.systemInfo              llama_print_system_info() of the loaded backends
//   model.tokenBos(), model.tokenEos()
//   model.tokenize(bytes|string, { addBos = false, special = false }) -> Int32Array
//       Same as llama-cpp-python's tokenize(): addBos maps to add_special and
//       special to parse_special, no extra leading-space handling on top of
//       what the vocabulary does itthis.
//   model.detokenize(tokens) -> Uint8Array
//       Same as llama-cpp-python's detokenize(): concatenates the raw pieces of
//       each token (special tokens render as nothing) and strips one leading
//       space only if the first token is BOS.
//   model.reset()                 clear the KV cache, nPast becomes 0
//   model.eval(tokens)            append tokens after the current position, in nBatch chunks
//   model.logits({ copy = true }) -> Float32Array, logits of the last evaluated token.
//       With copy false the array aliases libllama memory and is only valid
//       until the next eval(), reset() or close().
//   model.close()                 free context and model, safe to call twice
//
// FFI notes
//   * llama_model_params, llama_context_params and llama_batch are passed and
//     returned by value. tjs:ffi supports that when the struct is described
//     with defineStruct(): the returned bytes are unpacked into an object, the
//     fields are edited, and the object is packed again for the next call.
//     Enums are i32 and C bool is bool_u8.
//   * The backends (CPU variants, Vulkan) are separate shared objects, so
//     ggml_backend_load_all() runs once before the first model load.
//   * Logging is replaced with a JSCallback that ignores everything (or
//     prints to stderr when verbose).
//   * Libraries are never dlclose()d. Unloading libllama while its OpenMP
//     threads are alive is not worth the risk, and the process is ending anyway.
import { dlopen, defineStruct, JSCallback, types, suffix } from 'tjs:ffi';

const IS_WINDOWS = suffix === 'dll';

// Versions whose llama.h layouts were checked, with their build numbers.
export const KNOWN_VERSIONS = { '1.0.0': 12359 };
const OLD_LIBRARY_MESSAGE = 'needs llama.cpp 1.0.0 (build 12359) or newer layout-compatible';

const libCandidates = () => {
    const names = suffix === 'dylib' ? ['libllama.dylib', 'libllama.1.dylib'] : IS_WINDOWS ? ['llama.dll', 'libllama.dll'] : ['libllama.so', 'libllama.so.1'];
    const dirs = suffix === 'dylib'
        ? ['/opt/homebrew/lib', '/usr/local/lib', '/opt/local/lib', '/usr/lib']
        : IS_WINDOWS
            ? ['']
            : ['/home/linuxbrew/.linuxbrew/lib', '/usr/local/lib', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib/aarch64-linux-gnu', '/lib/x86_64-linux-gnu'];
    // Bare names last, so the dynamic loader's own search path still gets a try.
    return [...names.flatMap(name => dirs.filter(Boolean).map(dir => `${dir}/${name}`)), ...names];
};

// Struct layouts for llama.h 1.0.0 (x86-64 and arm64 LP64: 80, 176 and 56 bytes).
// defineStruct() inserts the alignment padding itself, the order is the header order.
export const ModelParams = defineStruct([
    ['devices', 'pointer'],
    ['tensor_buft_overrides', 'pointer'],
    ['n_gpu_layers', 'i32'],
    ['split_mode', 'i32'],
    ['load_mode', 'i32'],
    ['lazy_mode', 'i32'],
    ['main_gpu', 'i32'],
    ['tensor_split', 'pointer'],
    ['progress_callback', 'pointer'],
    ['progress_callback_user_data', 'pointer'],
    ['kv_overrides', 'pointer'],
    ['vocab_only', 'bool_u8'],
    ['check_tensors', 'bool_u8'],
    ['use_extra_bufts', 'bool_u8'],
    ['no_host', 'bool_u8'],
    ['no_alloc', 'bool_u8'],
    ['load_mtp', 'bool_u8'],
]);
export const ContextParams = defineStruct([
    ['n_ctx', 'u32'],
    ['n_batch', 'u32'],
    ['n_ubatch', 'u32'],
    ['n_seq_max', 'u32'],
    ['n_rs_seq', 'u32'],
    ['n_outputs_max', 'u32'],
    ['n_outputs_max_per_seq', 'u32'],
    ['n_threads', 'i32'],
    ['n_threads_batch', 'i32'],
    ['ctx_type', 'i32'],
    ['rope_scaling_type', 'i32'],
    ['pooling_type', 'i32'],
    ['attention_type', 'i32'],
    ['flash_attn_type', 'i32'],
    ['rope_freq_base', 'f32'],
    ['rope_freq_scale', 'f32'],
    ['yarn_ext_factor', 'f32'],
    ['yarn_attn_factor', 'f32'],
    ['yarn_beta_fast', 'f32'],
    ['yarn_beta_slow', 'f32'],
    ['yarn_orig_ctx', 'u32'],
    ['cb_eval', 'pointer'],
    ['cb_eval_user_data', 'pointer'],
    ['type_k', 'i32'],
    ['type_v', 'i32'],
    ['moe_cache_size', 'u64'],
    ['moe_cache_cpu_share', 'f32'],
    ['abort_callback', 'pointer'],
    ['abort_callback_data', 'pointer'],
    ['embeddings', 'bool_u8'],
    ['offload_kqv', 'bool_u8'],
    ['no_perf', 'bool_u8'],
    ['op_offload', 'bool_u8'],
    ['swa_full', 'bool_u8'],
    ['kv_unified', 'bool_u8'],
    ['samplers', 'pointer'],
    ['n_samplers', 'u64'],
    ['ctx_other', 'pointer'],
]);
export const Batch = defineStruct([
    ['n_tokens', 'i32'],
    ['token', 'pointer'],
    ['embd', 'pointer'],
    ['pos', 'pointer'],
    ['n_seq_id', 'pointer'],
    ['seq_id', 'pointer'],
    ['logits', 'pointer'],
]);

const LOAD_MODE_MMAP = 1;
const LOAD_MODE_MLOCK = 2;

let native = null;
let backendUsers = 0;
let logCallback = null;
let logVerbose = false;
let logTail = [];

const symbolTable = {
    llama_version: { args: [], returns: 'string', optional: true },
    llama_print_system_info: { args: [], returns: 'string' },
    llama_backend_init: { args: [], returns: 'void' },
    llama_backend_free: { args: [], returns: 'void' },
    llama_log_set: { args: ['pointer', 'pointer'] },
    llama_model_default_params: { returns: ModelParams },
    llama_context_default_params: { returns: ContextParams },
    llama_model_load_from_file: { args: ['string', ModelParams], returns: 'pointer' },
    llama_model_free: { args: ['pointer'] },
    llama_init_from_model: { args: ['pointer', ContextParams], returns: 'pointer' },
    llama_free: { args: ['pointer'] },
    llama_n_ctx: { args: ['pointer'], returns: 'u32' },
    llama_n_batch: { args: ['pointer'], returns: 'u32' },
    llama_model_get_vocab: { args: ['pointer'], returns: 'pointer' },
    llama_vocab_n_tokens: { args: ['pointer'], returns: 'i32' },
    llama_vocab_bos: { args: ['pointer'], returns: 'i32' },
    llama_vocab_eos: { args: ['pointer'], returns: 'i32' },
    llama_get_memory: { args: ['pointer'], returns: 'pointer' },
    llama_memory_clear: { args: ['pointer', 'bool_u8'] },
    llama_batch_get_one: { args: ['buffer', 'i32'], returns: Batch },
    llama_decode: { args: ['pointer', Batch], returns: 'i32' },
    llama_get_logits_ith: { args: ['pointer', 'i32'], returns: 'pointer' },
    llama_tokenize: { args: ['pointer', 'buffer', 'i32', 'buffer', 'i32', 'bool_u8', 'bool_u8'], returns: 'i32' },
    llama_token_to_piece: { args: ['pointer', 'i32', 'buffer', 'i32', 'i32', 'bool_u8'], returns: 'i32' },
};

// llama_build_number() is C++ and lives in libllama-common, which not every
// package ships. The version table covers the known releases without it.
const readBuild = libPath => {
    const dir = libPath.includes('/') ? libPath.slice(0, libPath.lastIndexOf('/') + 1) : '';
    const names = suffix === 'dylib' ? ['libllama-common.dylib', 'libllama-common.1.dylib'] : IS_WINDOWS ? ['llama-common.dll'] : ['libllama-common.so', 'libllama-common.so.1'];
    for (const name of [...names.map(base => dir + base), ...names]) {
        try {
            const symbol = IS_WINDOWS ? '?llama_build_number@@YAHXZ' : '_Z18llama_build_numberv';
            const number = dlopen(name, { [symbol]: { args: [], returns: 'i32' } }).symbols[symbol]();
            if (number > 0) return number;
        }
        catch {}
    }
    return null;
};

export const checkVersion = (llamaVersion, libPath) => {
    if (typeof llamaVersion !== 'function') throw new Error(`${libPath} ${OLD_LIBRARY_MESSAGE}. It has no llama_version(), so its struct layouts predate the ones this binding uses.`);
    const version = String(llamaVersion());
    if (!Object.hasOwn(KNOWN_VERSIONS, version) && tjs.env.LLAMA_ZIP_ALLOW_UNTESTED !== '1') {
        throw new Error(`${libPath} is llama.cpp ${version}, which llama-zip.js has not been tested with (tested: ${Object.keys(KNOWN_VERSIONS).join(', ')}). `
            + 'A changed struct layout corrupts calls without any error. Set LLAMA_ZIP_ALLOW_UNTESTED=1 to use it anyway.');
    }
    return { version, build: KNOWN_VERSIONS[version] ?? readBuild(libPath) };
};

const loadNative = () => {
    if (native) return native;
    const candidates = [];
    if (tjs.env.LLAMA_LIB) candidates.push(tjs.env.LLAMA_LIB);
    candidates.push(...libCandidates());
    const failures = [];
    let lib = null;
    let libPath = null;
    for (const candidate of candidates) {
        try {
            lib = dlopen(candidate, symbolTable);
            libPath = candidate;
            break;
        }
        catch (error) {
            failures.push(`${candidate}: ${error?.message ?? error}`);
        }
    }
    if (!lib) {
        throw new Error(`libllama not found. Set LLAMA_LIB to its full path. Tried (dlopen errors): ${failures.join('; ')}`);
    }
    const { version, build } = checkVersion(lib.symbols.llama_version, libPath);
    // The CPU and GPU backends are separate shared objects. ggml_backend_load_all lives in libggml-base,
    // which libllama already loaded, so asking for it by name returns the same handle.
    const dir = libPath.includes('/') ? libPath.slice(0, libPath.lastIndexOf('/') + 1) : '';
    const baseNames = suffix === 'dylib' ? ['libggml-base.dylib'] : IS_WINDOWS ? ['ggml-base.dll'] : ['libggml-base.so', 'libggml-base.so.0'];
    let ggml = {};
    for (const name of [...baseNames.map(base => dir + base), ...baseNames]) {
        try {
            ggml = dlopen(name, { ggml_backend_load_all: { args: [], returns: 'void', optional: true } }).symbols;
            break;
        }
        catch {}
    }
    native = { lib, symbols: lib.symbols, ggml, libPath, version, build };
    return native;
};

// The version and build of the library that LlamaModel would use, { version, build, text }.
// Throws what the constructor throws when there is no usable library. The app uses it to
// name the installed version in its messages after a model failed to load.
export const libraryInfo = () => {
    const { version, build } = loadNative();
    return { version, build, text: build ? `${version} (build ${build})` : version };
};

const installLogging = (symbols, verbose) => {
    logVerbose = verbose;
    if (logCallback) return;
    // The callback outlives every model, so a native thread logging late never hits freed memory.
    // A void JSCallback throws "cannot convert js val to void" on return, so it is declared as int and returns 0.
    // The C caller expects void and ignores the register.
    logCallback = new JSCallback(types.sint32, [types.sint32, types.string, types.pointer], (level, text) => {
        if (logVerbose) console.error(text.replace(/\n$/, ''));
        else if (level >= 3) {
            logTail.push(text.trim());
            if (logTail.length > 5) logTail.shift();
        }
        return 0;
    });
    symbols.llama_log_set(logCallback.addr, null);
};

const toBytes = data => typeof data === 'string' ? new TextEncoder().encode(data) : data;

const toInt32 = tokens => tokens instanceof Int32Array ? tokens : Int32Array.from(tokens);

export class LlamaModel {
    #n;
    #model = null;
    #ctx = null;
    #vocab = null;
    #nBatch;
    #nPast = 0;
    #hasLogits = false;
    #logitsBuf = null;

    nCtx = 0;
    nVocab = 0;
    llamaVersion = '';
    llamaBuild = '';
    systemInfo = '';

    constructor({ path: modelPath, nCtx = 0, nGpuLayers = 0, useMlock = false, nBatch = 512, nThreads = 0, verbose = false } = {}) {
        if (!modelPath) throw new Error('LlamaModel: path is required');
        const n = loadNative();
        this.#n = n;
        installLogging(n.symbols, verbose);
        logTail = [];
        if (backendUsers++ === 0) {
            n.symbols.llama_backend_init();
            n.ggml.ggml_backend_load_all?.();
        }
        this.llamaVersion = n.version;
        this.llamaBuild = n.build ? `${n.version} (build ${n.build})` : n.version;
        this.systemInfo = n.symbols.llama_print_system_info();
        try {
            const mp = n.symbols.llama_model_default_params();
            mp.n_gpu_layers = nGpuLayers;
            mp.load_mode = useMlock ? LOAD_MODE_MMAP | LOAD_MODE_MLOCK : LOAD_MODE_MMAP;
            this.#model = n.symbols.llama_model_load_from_file(modelPath, mp);
            if (!this.#model) throw new Error(`LlamaModel: failed to load model ${modelPath}. ${logTail.join(' ')}`.trim());
            const cp = n.symbols.llama_context_default_params();
            cp.n_ctx = nCtx;
            cp.n_batch = nBatch;
            cp.n_ubatch = Math.min(cp.n_ubatch, nBatch);
            if (nThreads > 0) {
                cp.n_threads = nThreads;
                cp.n_threads_batch = nThreads;
            }
            if (nGpuLayers === 0) {
                // Keep every op on the CPU so results do not depend on a GPU being present.
                cp.offload_kqv = false;
                cp.op_offload = false;
            }
            this.#ctx = n.symbols.llama_init_from_model(this.#model, cp);
            if (!this.#ctx) throw new Error(`LlamaModel: failed to create context. ${logTail.join(' ')}`.trim());
            this.#vocab = n.symbols.llama_model_get_vocab(this.#model);
            // llama.cpp clamps n_batch to the context size and aborts the process on a larger batch, so ask it.
            this.#nBatch = n.symbols.llama_n_batch(this.#ctx);
            this.nCtx = n.symbols.llama_n_ctx(this.#ctx);
            this.nVocab = n.symbols.llama_vocab_n_tokens(this.#vocab);
            this.#logitsBuf = new Float32Array(this.nVocab);
        }
        catch (error) {
            this.close();
            throw error;
        }
    }

    get nBatch() {
        return this.#nBatch;
    }

    get nPast() {
        return this.#nPast;
    }

    #assertOpen() {
        if (!this.#ctx) throw new Error('LlamaModel is closed');
    }

    tokenBos() {
        this.#assertOpen();
        return this.#n.symbols.llama_vocab_bos(this.#vocab);
    }

    tokenEos() {
        this.#assertOpen();
        return this.#n.symbols.llama_vocab_eos(this.#vocab);
    }

    tokenize(data, { addBos = false, special = false } = {}) {
        this.#assertOpen();
        const text = toBytes(data);
        const textArg = text.length ? text : new Uint8Array(1);
        let out = new Int32Array(text.length + (addBos ? 1 : 0) + 8);
        let count = this.#n.symbols.llama_tokenize(this.#vocab, textArg, text.length, new Uint8Array(out.buffer), out.length, addBos, special);
        if (count < 0) {
            out = new Int32Array(-count);
            count = this.#n.symbols.llama_tokenize(this.#vocab, textArg, text.length, new Uint8Array(out.buffer), out.length, addBos, special);
        }
        if (count < 0) throw new Error(`LlamaModel.tokenize failed (${count})`);
        return out.slice(0, count);
    }

    detokenize(tokens) {
        this.#assertOpen();
        const list = Array.from(tokens);
        const pieces = [];
        let piece = new Uint8Array(32);
        let total = 0;
        for (const token of list) {
            let length = this.#n.symbols.llama_token_to_piece(this.#vocab, token, piece, piece.length, 0, false);
            if (length < 0) {
                piece = new Uint8Array(-length);
                length = this.#n.symbols.llama_token_to_piece(this.#vocab, token, piece, piece.length, 0, false);
            }
            if (length < 0) throw new Error(`LlamaModel.detokenize failed for token ${token}`);
            pieces.push(piece.slice(0, length));
            total += length;
        }
        const output = new Uint8Array(total);
        let offset = 0;
        for (const part of pieces) {
            output.set(part, offset);
            offset += part.length;
        }
        if (list.length > 0 && list[0] === this.tokenBos() && output[0] === 0x20) return output.slice(1);
        return output;
    }

    reset() {
        this.#assertOpen();
        this.#n.symbols.llama_memory_clear(this.#n.symbols.llama_get_memory(this.#ctx), true);
        this.#nPast = 0;
        this.#hasLogits = false;
    }

    eval(tokens) {
        this.#assertOpen();
        const all = toInt32(tokens);
        if (all.length === 0) return;
        if (this.#nPast + all.length > this.nCtx) {
            throw new Error(`LlamaModel.eval: ${this.#nPast} + ${all.length} tokens exceed the context of ${this.nCtx}`);
        }
        for (let start = 0; start < all.length; start += this.#nBatch) {
            // A copy per chunk, because llama_batch_get_one keeps a pointer into this buffer.
            const chunk = all.slice(start, start + this.#nBatch);
            const batch = this.#n.symbols.llama_batch_get_one(new Uint8Array(chunk.buffer), chunk.length);
            const status = this.#n.symbols.llama_decode(this.#ctx, batch);
            if (status !== 0) {
                this.#hasLogits = false;
                throw new Error(`llama_decode failed with status ${status}`);
            }
            this.#nPast += chunk.length;
        }
        this.#hasLogits = true;
    }

    logits({ copy = true } = {}) {
        this.#assertOpen();
        if (!this.#hasLogits) throw new Error('LlamaModel.logits: nothing evaluated yet');
        const pointer = this.#n.symbols.llama_get_logits_ith(this.#ctx, -1);
        if (!pointer) throw new Error('LlamaModel.logits: libllama returned no logits');
        const view = new Float32Array(pointer.toArrayBuffer(this.nVocab * 4));
        if (!copy) return view;
        this.#logitsBuf.set(view);
        return this.#logitsBuf.slice();
    }

    close() {
        if (!this.#n) return;
        if (this.#ctx) this.#n.symbols.llama_free(this.#ctx);
        if (this.#model) this.#n.symbols.llama_model_free(this.#model);
        this.#ctx = null;
        this.#model = null;
        this.#vocab = null;
        this.#hasLogits = false;
        if (--backendUsers === 0) this.#n.symbols.llama_backend_free();
        this.#n = null;
    }
}
