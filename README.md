# llama-zip

llama-zip is a lossless compression utility that leverages a user-provided LLM (large language model) as the probabilistic model for an [arithmetic coder](https://en.wikipedia.org/wiki/Arithmetic_coding). This allows llama-zip to achieve high compression ratios on structured or natural language text, since few bits are needed to encode tokens that the model predicts with high confidence. By employing a sliding context window, llama-zip is not limited by the context length of the LLM and can compress strings of arbitrary length. Furthermore, by encoding invalid UTF-8 bytes using code points in Unicode's private use areas, llama-zip is not limited to text inputs and can handle arbitrary binary data, albeit with reduced compression ratios compared to text inputs.

![Interactive Mode Demo: Lorem Ipsum Text](lorem_ipsum_demo.gif)

## Compression Performance

In the table below, the compression ratios achieved by llama-zip on the text files of the [Calgary Corpus](http://www.data-compression.info/Corpora/CalgaryCorpus/) (as well as on an earlier version of llama-zip's own source code, `llama_zip.py`) are compared with other popular and/or high-performance compression utilities. Compression ratios are calculated by dividing the number of bytes in the uncompressed input by the number of bytes in the compressed output, meaning that higher values indicate more effective compression.

For llama-zip, three LLMs were benchmarked at two context lengths, with a consistent window overlap of 25% (see [Options](#options) for more info):

- [Qwen3.5-35B-A3B-Base (Q8_0)](https://huggingface.co/mradermacher/Qwen3.5-35B-A3B-Base-GGUF)
  - 8192-token context length ("qwen-8k" in the table)
  - 32768-token context length ("qwen-32k" in the table)
- [Phi-3.1-mini-128k-instruct (Q4_K_M)](https://huggingface.co/bartowski/Phi-3.1-mini-128k-instruct-GGUF)
  - 8192-token context length ("phi-8k" in the table)
  - 32768-token context length ("phi-32k" in the table)
- [Llama-3.1-8B (Q4_K_M)](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-GGUF)
  - 8192-token context length ("llama-8k" in the table)
  - 32768-token context length ("llama-32k" in the table)

For the other utilities, the maximum compression level offered was used.

| File         | llama&#8209;zip (qwen&#8209;32k) | llama&#8209;zip (qwen&#8209;8k) | llama&#8209;zip (llama&#8209;8k) | llama&#8209;zip (llama&#8209;32k) | llama&#8209;zip (phi&#8209;32k) | llama&#8209;zip (phi&#8209;8k) |  cmix | paq8px | paq8pxd |  zpaq | brotli | bzip2 |  lzma |    xz |  zstd |  gzip |
| :----------- | -------------------------------: | ------------------------------: | -------------------------------: | --------------------------------: | ------------------------------: | -----------------------------: | ----: | -----: | ------: | ----: | -----: | ----: | ----: | ----: | ----: | ----: |
| bib          |                       **15.520** |                          14.851 |                <ins>15.001</ins> |                            13.577 |                          10.473 |                          9.920 | 5.633 |  5.668 |   5.590 | 4.611 |  3.920 | 4.051 | 3.641 | 3.636 | 3.485 | 3.171 |
| book1        |                 <ins>8.610</ins> |                           8.497 |                        **8.745** |                             8.278 |                           7.087 |                          6.997 | 4.209 |  4.192 |   4.204 | 3.823 |  2.999 | 3.305 | 2.942 | 2.941 | 2.904 | 2.460 |
| book2        |                       **13.766** |               <ins>13.055</ins> |                           12.250 |                            11.852 |                          10.682 |                         10.108 | 5.381 |  5.346 |   5.325 | 4.649 |  3.696 | 3.880 | 3.598 | 3.596 | 3.514 | 2.963 |
| news         |                       **10.488** |               <ins>10.107</ins> |                            9.976 |                             9.350 |                           8.395 |                          7.935 | 4.542 |  4.531 |   4.494 | 3.817 |  3.338 | 3.180 | 3.173 | 3.171 | 3.073 | 2.610 |
| paper1       |                       **14.763** |               <ins>14.233</ins> |                           12.577 |                            12.869 |                          10.442 |                         10.072 | 4.264 |  4.302 |   4.212 | 3.572 |  3.439 | 3.211 | 3.083 | 3.074 | 3.017 | 2.867 |
| paper2       |                       **13.228** |               <ins>12.949</ins> |                           12.370 |                            12.460 |                          10.787 |                         10.561 | 4.180 |  4.208 |   4.135 | 3.679 |  3.308 | 3.283 | 3.020 | 3.015 | 2.982 | 2.769 |
| progc        |                       **20.992** |               <ins>20.429</ins> |                           13.802 |                            14.002 |                          10.714 |                         10.188 | 4.439 |  4.438 |   4.352 | 3.495 |  3.409 | 3.158 | 3.162 | 3.151 | 3.096 | 2.968 |
| progl        |                       **23.506** |               <ins>22.093</ins> |                           20.429 |                            20.228 |                          14.733 |                         14.054 | 7.497 |  7.464 |   7.347 | 5.554 |  5.116 | 4.599 | 4.801 | 4.787 | 4.728 | 4.432 |
| progp        |                       **26.182** |               <ins>22.882</ins> |                           20.438 |                            21.888 |                          16.722 |                         15.450 | 7.705 |  7.665 |   7.508 | 5.348 |  4.998 | 4.611 | 4.792 | 4.772 | 4.724 | 4.414 |
| trans        |                       **17.008** |               <ins>14.244</ins> |                           12.523 |                            13.497 |                          11.746 |                          9.776 | 8.650 |  8.484 |   8.409 | 6.597 |  6.083 | 5.235 | 5.628 | 5.613 | 5.417 | 4.949 |
| llama_zip.py |                       **38.178** |                      **38.178** |                <ins>29.083</ins> |                 <ins>29.083</ins> |                          23.383 |                         23.383 | 4.904 |  4.976 |   4.689 | 3.018 |  3.980 | 3.508 | 3.608 | 3.552 | 3.633 | 3.542 |

The best-performing compressor for each file is listed in bold, and the second-best is underlined. The columns are sorted by average compression ratio achieved across all files, with overall better-performing compressors listed further to the left.

These results show that llama-zip can significantly outperform traditional compression utilities, at least with the LLMs and files tested. Note that increased context length might not lead to better compression ratios past a certain point, though, as evidenced by the fact that Llama 3.1 performed better on average with an 8k-token context length than with a 32k-token context length. Context lengths longer than 32k tokens were not tested due to memory and time constraints.

## Installation

```sh
git clone https://github.com/alexbuz/llama-zip.git
cd llama-zip
pip3 install .
```

### LLM Download

To use llama-zip, you must download an LLM that is compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp), such as [Llama 3.1 8B](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-GGUF). Make sure to download a quantized version (one of the `.gguf` files listed on the "Files and versions" tab on Hugging Face) that is small enough to fit in your system's memory.

## CLI Usage

```
llama-zip <llm_path> [options] <mode> [input]
```

### Modes

llama-zip supports three modes of operation:

1. **Compress mode** (specified by the `-c` or `--compress` flag): The string to be compressed can be provided as an argument or piped to stdin. The compressed output will be written to stdout.
2. **Decompress mode** (specified by the `-d` or `--decompress` flag): The compressed string can be provided as an argument or piped to stdin. The decompressed output will be written to stdout.
3. **Interactive mode** (specified by the `-i` or `--interactive` flag): A prompt is displayed where the user can enter strings to be compressed or decompressed. When a base64-encoded string is entered, it will be treated as representing compressed data and will be decompressed; otherwise, it will be compressed. After each compression or decompression operation, the user is prompted to enter another string. To exit interactive mode, press `Ctrl+C`.
   - **Note:** If you would like to compress a string that consists entirely of base64 characters (i.e., letters, numbers, `+`, and `/`, without any other symbols or spaces), you must use compression mode directly, as interactive mode assumes that base64-encoded strings are meant to be decompressed and will result in nonsensical output if the input did not come from a compression operation. Alternatively, you can add a non-base64 character to your string (such as a space at the end) if you don't mind that character being compressed along with the rest of the string.

### Options

- `-f`, `--compressed-format`: The format of compressed data. This can be set to `binary` (the default for non-interactive modes) or `base64` (the default and only supported format for interactive mode).
- `-w`, `--window-overlap`: The number of tokens to overlap between the end of the previous context window and the start of the next window, when compressing a string whose length exceeds the model's maximum context length. This can be specified as a percentage of the model's context length or as a fixed number of tokens. The default is `0%`, meaning that the context window is cleared entirely when it is filled. Higher values can improve compression ratios but will slow down compression and decompression. Note that when decompressing, the window overlap must be set to the same value that was used during compression in order to reconstruct the original string.
- `--n-ctx`: The number of tokens to use as the context length for the model. This must be less than or equal to the model's maximum context length. If set to `0` (the default), then the model's maximum context length will be used. Note that when decompressing, the context length must be set to the same value that was used during compression in order to reconstruct the original string.
- `--n-gpu-layers`: The number of model layers to offload to the GPU. This can significantly speed up compression and decompression, especially for larger models. If set to `-1` (the default), then all layers will be offloaded. See the [llama.cpp repository](https://github.com/ggerganov/llama.cpp) for more information. In [practice](#limitations), the same number of layers should be offloaded during compression and decompression.
- `--use-mlock`: Force your system to keep the entire model in memory. This can be useful for larger models but may cause your system to run out of memory if the model is too large. Disabled by default.

### Examples

#### Compression

- Compress a file:

  ```sh
  llama-zip /path/to/llm.gguf -c < input.txt > compressed.llzp
  ```

- Compress a string and print the compressed output in base64 format:
  ```sh
  llama-zip /path/to/llm.gguf -f base64 -c "The quick brown fox jumps over the lazy dog."
  ```

#### Decompression

- Decompress a file:

  ```sh
  llama-zip /path/to/llm.gguf -d < compressed.llzp > output.txt
  ```

- Decompress a base64-encoded compressed string:
  ```sh
  llama-zip /path/to/llm.gguf -f base64 -d BASE64_STRING
  ```

#### Interactive Mode

- Start an interactive mode session:
  ```sh
  llama-zip /path/to/llm.gguf -i
  ```

#### Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKL-ZVucgbVcZnEi9NyfjMIJ_PrTKMEW?usp=sharing)

## API Usage

The `LlamaZip` class can be used to compress and decompress data programmatically. The `compress` method takes a `bytes` object and returns a another `bytes` object containing the compressed data. The `decompress` method takes a `bytes` object containing compressed data and returns the original uncompressed data.

```python
from llama_zip import LlamaZip

# Initialize the compressor
compressor = LlamaZip(model_path="/path/to/model.gguf")

# Compress some data
original = b"The quick brown fox jumps over the lazy dog."
compressed = compressor.compress(original)
assert len(compressed) < len(original)

# Decompress the data
decompressed = compressor.decompress(compressed)
assert decompressed == original
```

The `LlamaZip` constructor also accepts the `n_ctx`, `n_gpu_layers`, and `use_mlock` arguments, which correspond to the CLI options of the same names. The `window_overlap` argument can be passed to the `compress` and `decompress` methods directly to specify the window overlap for that particular operation.

## Limitations

1. **Speed:** Compression and decompression speeds are limited by the speed of LLM inference. This renders llama-zip significantly slower than traditional compression utilities. However, the compression ratios achieved by llama-zip may justify the trade-off in speed for certain use cases.
2. **Portability:** llama-zip requires identical LLM behavior during compression and decompression. However, the backend that llama-zip uses for LLM inference, [llama.cpp](https://github.com/ggerganov/llama.cpp), does not currently guarantee deterministic behavior. This limits the portability of the compressed output of llama-zip, as it may not be decompressible on a different system, even if the same model is used. In practice, behavior also differs depending on the number of GPU layers offloaded, so the `--n-gpu-layers` option should be set to the same value during compression and decompression, in addition to the window overlap (`--window-overlap`) and context length (`--n-ctx`) options.
3. **Binary Compression:** Due to its reliance on an LLM for prediction, llama-zip is best suited for compressing inputs that consist primarily of text. Although llama-zip can handle binary data by encoding invalid UTF-8 bytes using code points in Unicode's private use areas, it may not achieve high compression ratios on such data, potentially producing compressed output that is larger than the original input.

## Tests

The Python unit tests live in `tests/python`. They replace `llama_cpp.Llama` with a deterministic fake model, so they run without llama-cpp-python or a model file.

```bash
pip install -r requirements-dev.txt
pytest --cov=llama_zip --cov-branch --cov-report=term-missing
```

If you only want the test dependencies without llama-cpp-python, install `numpy`, `more-itertools`, `tqdm`, `pytest` and `pytest-cov`.

Model-independent test vectors (arithmetic coder, UTF-8 mapping, base64, window overlap, `compute_cdf` and fake-model round trips) are stored in `tests/vectors/*.json` so other implementations can reuse them. `python tests/python/generate_vectors.py` regenerates them from `llama_zip.py`.

`tests/python/test_integration.py` does a real compress and decompress round trip. It is skipped unless `llama_cpp` is importable and the model file exists. Set `LLAMA_ZIP_TEST_MODEL` to the path of a GGUF file to choose the model.

## JavaScript port (txiki.js)

`js/` holds a port of `llama_zip.py` to [txiki.js](https://github.com/saghul/txiki.js) (`tjs`). It calls the system libllama through `tjs:ffi`, so it needs no Python.

| File | Contents |
| --- | --- |
| `js/llamazip.js` | `LlamaZip` class (`compress`, `decompress`, `computeCdf`) |
| `js/arithmetic.js` | `Encoder` and `Decoder`, with BigInt for the 64-bit state |
| `js/utf8.js` | `Utf8Chunks`, `bytesToUtf8`, `utf8ToBytes` and the private use area mapping |
| `js/cli.js` | Command line tool with the same flags as `llama_zip.py` |
| `js/llama.js` | libllama binding (`LlamaModel`) |
| `js/io.js` | stdout and stderr sinks |

### Requirements

- `tjs` v26.11.0 or newer.
- libllama from llama.cpp build b10180 (the struct layouts in `js/llama.js` are checked at load time, so another ABI fails with an error). Install it with `brew install llama.cpp`, or `winget install ggml.llamacpp` on Windows. If the library is not in a standard location, set `LLAMA_LIB` to its full path.
- A GGUF model, as for the Python tool.

### Command line

The options are the ones of the Python tool, plus `--n-threads`. `-h` lists them.

```sh
# Compress an argument string, print base64
tjs run js/cli.js /path/to/llm.gguf -f base64 -c "The quick brown fox jumps over the lazy dog."

# Compress a file to binary, then decompress it
tjs run js/cli.js /path/to/llm.gguf -c < file.txt > file.lz
tjs run js/cli.js /path/to/llm.gguf -d < file.lz

# Interactive mode, with a window overlap of 50% of the context
tjs run js/cli.js /path/to/llm.gguf -i -w 50%
```

Ctrl-C during compression stops early and finishes the stream, so the output still decompresses to the part that was read. During decompression and at the prompt it exits like the Python tool.

### Library

```js
import { LlamaZip } from './js/llamazip.js';

const zip = new LlamaZip('/path/to/model.gguf', { nCtx: 0, nGpuLayers: -1, nThreads: 4 });
const compressed = await zip.compress(new TextEncoder().encode('The quick brown fox.'), 0);
const original = await zip.decompress(compressed, 0);
zip.close();
```

`compress` and `decompress` return `Uint8Array` and are async, so the event loop runs while they work. Both take `{ signal }` (an `AbortSignal`) as a third argument, and `decompress` also takes `{ maxTokens }`. `decompress` throws an error when the data ends without an end-of-stream token, where the Python tool would loop forever.

### Differences from the Python tool

- Compressed data from the two tools is not interchangeable. The Python tool computes the log-softmax in float32 with numpy, whose SIMD `exp` and `log` cannot be reproduced bit for bit in JavaScript. The JS port computes it in float64 and rounds half to even like `numpy.round`. The two cumulative frequency tables differ by a few units out of 2^32 for some tokens, and an arithmetic decoder diverges after that. For the sentence "The quick brown fox jumps over the lazy dog." the outputs already differ in the second character. The model logits themselves were bit-identical for a single token context.
- Compression and decompression must use the same `--n-threads` (default 4, fixed so it does not depend on the machine), the same window overlap and `--n-ctx`, and the same libllama build. The format stores none of this. Each window is evaluated as `[BOS]` plus the overlap tokens, then one token per call, like `generate()` in llama-cpp-python, but without its reuse of the previous window's cache.
- It is slower: about 45 tokens per second against about 100 for the Python tool on SmolLM2-135M Q8_0 (16 cores, CPU only). The model itself takes the same time. The remainder is the softmax over the 49,152 vocabulary entries, which txiki.js runs in an interpreter. Larger models spend relatively less time there.
- A window overlap must be in `[0, n_ctx - 1]`, or the call throws.
- The tokenizer of some models drops bytes. With SmolLM2-135M the bytes 0x04, 0x06, 0x13, 0x14, 0x16 and 0x1d do not survive tokenizing and detokenizing, so data containing them does not round trip. The Python tool has the same problem.

### Tests

The JS tests load the same vectors as the Python tests (`tests/vectors/*.json`) and use a fake model that follows `tests/vectors/llamazip_fake_model.json`, so everything except the real-model tests runs without libllama.

```sh
tjs run js/test/all.js                      # every module, one process each
tjs run js/test/all.js arithmetic utf8      # chosen modules
tjs run js/test/llamazip.test.js            # one module
```

The modules are `arithmetic`, `utf8`, `base64`, `compute_cdf`, `llamazip` and `cli`, which mirror the Python modules of the same names, `integration`, which does round trips with the real model and the CLI and is skipped when libllama or the model is missing (`LLAMA_ZIP_TEST_MODEL` and `LLAMA_LIB` choose them), and `llama`, the binding's own tests. txiki.js has no coverage tool, so the JS suite has no coverage number.
