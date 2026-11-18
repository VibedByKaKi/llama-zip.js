"""Regenerate tests/vectors/*.json by running llama_zip.py.

Run from the repository root:  python tests/python/generate_vectors.py
The vectors are model independent, so another implementation can load the
same files. Review the diff after regenerating, since they are the contract.
"""

import json
import random
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "tests" / "vectors"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import fake_llama  # noqa: E402

stub = types.ModuleType("llama_cpp")
stub.Llama = fake_llama.FakeLlama
sys.modules["llama_cpp"] = stub

import llama_zip  # noqa: E402


def write(name, data):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=True)
        f.write("\n")


class CountingEncoder(llama_zip.Encoder):
    def __init__(self):
        super().__init__()
        self.max_underflow = 0

    def underflow(self):
        super().underflow()
        self.max_underflow = max(self.max_underflow, self.num_underflow)


def arithmetic_case(name, cdfs, steps, note=None):
    encoder = CountingEncoder()
    for cdf_index, symbol in steps:
        encoder.encode_symbol(cdfs[cdf_index], symbol)
    encoder.finish()
    encoded = bytes(encoder.get_encoded())
    decoder = llama_zip.Decoder(encoded)
    decoded = [int(decoder.decode_symbol(cdfs[i])) for i, _ in steps]
    assert decoded == [s for _, s in steps], name
    case = {
        "name": name,
        "cdfs": cdfs,
        "steps": [list(s) for s in steps],
        "encoded_hex": encoded.hex(),
        "max_underflow": encoder.max_underflow,
    }
    if note:
        case["note"] = note
    return case


def cdf_from_freqs(freqs):
    out, total = [], 0
    for f in freqs:
        total += f
        out.append(total)
    return out


def random_cdf(rng, size, total):
    cuts = sorted(rng.sample(range(1, total), size - 1)) + [total]
    return cuts


def arithmetic_vectors():
    rng = random.Random(1234)
    uniform256 = list(range(1, 257))
    cases = [
        arithmetic_case("empty message", [[1, 2]], [], "finish() alone writes one 1 bit"),
        arithmetic_case("single symbol alphabet", [[1]], [[0, 0]]),
        arithmetic_case("single symbol alphabet repeated", [[1]], [[0, 0]] * 40),
        arithmetic_case("binary first symbol", [[1, 2]], [[0, 0]]),
        arithmetic_case("binary second symbol", [[1, 2]], [[0, 1]]),
        arithmetic_case("binary alternating", [[1, 2]], [[0, i % 2] for i in range(20)]),
        arithmetic_case("uniform 256 all bytes", [uniform256], [[0, i] for i in range(256)]),
        arithmetic_case("uniform 256 first and last", [uniform256], [[0, 0], [0, 255], [0, 0], [0, 255]]),
        arithmetic_case(
            "skewed three symbols",
            [[8, 9, 10]],
            [[0, s] for s in (0, 0, 1, 0, 2, 0, 0, 0, 1, 2)],
        ),
        arithmetic_case(
            "underflow middle symbol repeated",
            [[1, 2, 3]],
            [[0, 1]] * 30,
            "the middle third straddles the half point, so every step underflows",
        ),
        arithmetic_case(
            "underflow then shift",
            [[1, 2, 3]],
            [[0, 1]] * 10 + [[0, 0]] * 3 + [[0, 1]] * 7 + [[0, 2]],
        ),
        arithmetic_case(
            "underflow with wide alphabet",
            [cdf_from_freqs([1] * 99 + [2] + [1] * 100)],
            [[0, 99]] * 25 + [[0, 3]],
        ),
        arithmetic_case(
            "large total at the 2^32 scale",
            [[1, (1 << 32) + 1]],
            [[0, 1], [0, 1], [0, 0], [0, 1]],
        ),
        arithmetic_case(
            "total 2^32 + vocabulary floor",
            [cdf_from_freqs([1 << 31, 1 << 30, 1 << 29, 1, 1, 1])],
            [[0, s] for s in (0, 1, 2, 3, 4, 5, 0, 5, 5, 4, 3)],
        ),
        arithmetic_case(
            "rare symbol with frequency 1 at 2^32 total",
            [[(1 << 32) - 1, 1 << 32]],
            [[0, 1]] * 6,
            "each step shrinks the range by about 2^-32",
        ),
        arithmetic_case(
            "rare first symbol at 2^32 total",
            [[1, 1 << 32]],
            [[0, 0]] * 6,
        ),
        arithmetic_case(
            "total 2^40",
            [[1, 3, 1 << 40]],
            [[0, s] for s in (2, 0, 1, 2, 2, 1, 0)],
        ),
        arithmetic_case(
            "total 2^52 needs 64-bit arithmetic",
            [[1 << 51, (1 << 51) + 1, 1 << 52]],
            [[0, s] for s in (1, 1, 1, 2, 0, 1, 2)],
        ),
        arithmetic_case(
            "zero width symbol is never encoded",
            [[5, 5, 9, 9, 10]],
            [[0, s] for s in (0, 2, 4, 2, 0)],
            "symbols 1 and 3 have zero frequency, the decoder must skip them",
        ),
        arithmetic_case(
            "alternating cdfs",
            [[1, 2], [3, 4, 5, 6], [1, 1000, 1001]],
            [[i % 3, (i * 7) % [2, 4, 3][i % 3]] for i in range(30)],
        ),
    ]
    big = random_cdf(rng, 50, (1 << 32) + 50)
    cases.append(
        arithmetic_case(
            "random 50 symbols near 2^32",
            [big],
            [[0, rng.randrange(50)] for _ in range(120)],
        )
    )
    cdfs = [random_cdf(rng, rng.randint(2, 30), rng.randint(40, 1 << 33)) for _ in range(6)]
    steps = []
    for _ in range(200):
        i = rng.randrange(len(cdfs))
        steps.append([i, rng.randrange(len(cdfs[i]))])
    cases.append(arithmetic_case("random mixed cdfs", cdfs, steps))
    found = 0
    for attempt in range(2000):
        cdf = cdf_from_freqs([rng.randint(1, 9) for _ in range(rng.randint(3, 6))])
        steps = [[0, rng.randrange(len(cdf))] for _ in range(rng.randint(20, 60))]
        case = arithmetic_case(f"random underflow search {attempt}", [cdf], steps)
        if case["max_underflow"] >= 2:
            case["name"] = f"random underflow {found}"
            cases.append(case)
            found += 1
            if found == 4:
                break
    assert found == 4

    decode_only = []
    cdf = [10, 30, 60, 100]
    for name, data in [
        ("empty input reads zero bits", b""),
        ("short input is zero padded", b"\x5a"),
        ("all ones input", b"\xff" * 12),
        ("alternating bits", b"\xaa\x55\xaa\x55\xaa\x55\xaa\x55\xaa\x55"),
        ("ten byte input", bytes(range(10, 20))),
    ]:
        decoder = llama_zip.Decoder(data)
        symbols = [int(decoder.decode_symbol(cdf)) for _ in range(12)]
        decode_only.append(
            {"name": name, "data_hex": data.hex(), "cdf": cdf, "symbols": symbols}
        )
    return {"cases": cases, "decode_only": decode_only}


def utf8_vectors():
    chunk_inputs = [
        ("empty", b""),
        ("ascii", b"hello"),
        ("two byte", "é".encode()),
        ("three byte", "€".encode()),
        ("four byte", "😀".encode()),
        ("mixed valid", "aé€😀z".encode()),
        ("lone continuation", b"\x80"),
        ("continuation between ascii", b"a\x80b"),
        ("invalid lead FF", b"\xff"),
        ("invalid lead F8", b"\xf8abc"),
        ("overlong C0 80", b"\xc0\x80"),
        ("overlong C1 BF", b"\xc1\xbf"),
        ("overlong E0 80 80", b"\xe0\x80\x80"),
        ("overlong E0 9F BF", b"\xe0\x9f\xbf"),
        ("overlong F0 80 80 80", b"\xf0\x80\x80\x80"),
        ("overlong F0 8F BF BF", b"\xf0\x8f\xbf\xbf"),
        ("surrogate ED A0 80", b"\xed\xa0\x80"),
        ("surrogate ED BF BF", b"\xed\xbf\xbf"),
        ("above U+10FFFF F4 90 80 80", b"\xf4\x90\x80\x80"),
        ("F5 lead", b"\xf5\x80\x80\x80"),
        ("max code point", "\U0010ffff".encode()),
        ("U+D7FF boundary", "퟿".encode()),
        ("U+E000 boundary", "".encode()),
        ("truncated 2 byte", b"\xc3"),
        ("truncated 3 byte after one", b"\xe2"),
        ("truncated 3 byte after two", b"\xe2\x82"),
        ("truncated 4 byte after one", b"\xf0"),
        ("truncated 4 byte after two", b"\xf0\x9f"),
        ("truncated 4 byte after three", b"\xf0\x9f\x98"),
        ("truncated then ascii", b"ok\xe2\x82z"),
        ("2 byte lead then ascii", b"\xc3A"),
        ("3 byte lead then ascii", b"\xe2A"),
        ("4 byte lead then ascii", b"\xf0\x9f\x98A"),
        ("valid then invalid then valid", b"a\xffb\xc3\xa9\x80"),
        ("many invalid", b"\xff\xfe\xfd\xfc"),
        ("PUA U+E000", b"\xee\x80\x80"),
        ("PUA U+E0FF", b"\xee\x83\xbf"),
        ("PUA U+E100", b"\xee\x84\x80"),
    ]
    chunk_cases, convert_cases = [], []
    for name, data in chunk_inputs:
        chunks = [
            {"valid_hex": c.valid.hex(), "invalid_hex": c.invalid.hex()}
            for c in llama_zip.Utf8Chunks(data)
        ]
        chunk_cases.append({"name": name, "input_hex": data.hex(), "chunks": chunks})
        out = llama_zip.bytes_to_utf8(data)
        convert_cases.append(
            {"name": name, "input_hex": data.hex(), "output_hex": out.hex()}
        )
        assert llama_zip.utf8_to_bytes(out.decode("utf-8")) == data, name

    to_bytes = []
    for name, text in [
        ("empty", ""),
        ("ascii", "abc"),
        ("non ASCII", "é€😀"),
        ("PUA start", ""),
        ("PUA end", ""),
        ("PUA mapped bytes", ""),
        ("past PUA range", ""),
        ("mixed", "ab"),
    ]:
        to_bytes.append(
            {
                "name": name,
                "codepoints": [ord(c) for c in text],
                "output_hex": llama_zip.utf8_to_bytes(text).hex(),
            }
        )

    rng = random.Random(99)
    roundtrip = [{"name": "all 256 byte values", "input_hex": bytes(range(256)).hex()}]
    roundtrip.append(
        {"name": "all 256 byte values reversed", "input_hex": bytes(range(255, -1, -1)).hex()}
    )
    for i in range(20):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 80)))
        roundtrip.append({"name": f"random bytes {i}", "input_hex": data.hex()})
    for i in range(10):
        pieces = []
        for _ in range(rng.randint(1, 20)):
            kind = rng.randrange(4)
            if kind == 0:
                pieces.append(rng.choice(["a", "é", "€", "😀", "", "", ""]).encode())
            elif kind == 1:
                pieces.append(bytes([rng.randrange(256)]))
            elif kind == 2:
                pieces.append(rng.choice(["€", "😀"]).encode()[: rng.randint(1, 2)])
            else:
                pieces.append(bytes(rng.randrange(0xE0, 0xE1 + 1) for _ in range(1)))
        roundtrip.append({"name": f"mixed pieces {i}", "input_hex": b"".join(pieces).hex()})
    for entry in roundtrip:
        data = bytes.fromhex(entry["input_hex"])
        assert llama_zip.utf8_to_bytes(llama_zip.bytes_to_utf8(data).decode("utf-8")) == data
    return {
        "chunks": chunk_cases,
        "bytes_to_utf8": convert_cases,
        "utf8_to_bytes": to_bytes,
        "roundtrip": roundtrip,
    }


def base64_vectors():
    inputs = [
        ("empty", b""),
        ("plain padded", b"aGVsbG8="),
        ("missing padding", b"aGVsbG8"),
        ("missing two padding", b"aGVsbA"),
        ("whitespace and newlines", b"aGVs\nbG8 =\r\n"),
        ("extra padding is dropped", b"aGVsbG8===="),
        ("padding in the middle", b"aGV=sbG8="),
        ("garbage characters dropped", b"a!G@V#s$b%G^8&"),
        ("url safe alphabet dropped", b"-_-_"),
        ("only invalid characters", b"!!!!****"),
        ("single character", b"A"),
        ("five characters", b"AAAAA"),
        ("non ASCII bytes dropped", b"aGVs\xffbG8="),
        ("all alphabet", llama_zip.BASE64),
        ("plus and slash", b"+/+/"),
        ("long", base64_encode(bytes(range(256)))),
        ("trailing bits nonzero", b"aGVsbG9="),
    ]
    return {
        "cases": [
            {
                "name": n,
                "input_hex": d.hex(),
                "output_hex": llama_zip.robust_b64decode(d).hex(),
            }
            for n, d in inputs
        ]
    }


def base64_encode(data):
    import base64

    return base64.b64encode(data)


def cdf_vectors():
    class Model:
        def logits_to_logprobs(self, logits):
            return fake_llama.FakeLlama.logits_to_logprobs(None, logits)

    zip_ = llama_zip.LlamaZip.__new__(llama_zip.LlamaZip)
    zip_.model = Model()
    cases = []
    for name, logits in [
        ("single token", [0.0]),
        ("uniform 2", [0.0, 0.0]),
        ("uniform 4", [1.0] * 4),
        ("uniform 256", [0.0] * 256),
        ("two to one", [1.0, 0.0]),
        ("one dominant token", [20.0, 0.0, 0.0, 0.0]),
        ("extreme gap hits the floor", [0.0, -100.0, -100.0, 0.0]),
        ("negative logits", [-1.5, -2.5, -0.5, -3.0]),
        ("shift invariance a", [1.0, 2.0, 3.0]),
        ("shift invariance b", [101.0, 102.0, 103.0]),
        ("monotone ramp", [i / 4 for i in range(16)]),
        ("tiny probabilities", [30.0] + [0.0] * 20),
        ("fake model logits", [float(x) for x in fake_llama.fake_logits([0, 5, 9])]),
    ]:
        cdf = zip_.compute_cdf(__import__("numpy").array(logits, dtype="float32"))
        cases.append(
            {"name": name, "logits": logits, "cum_freqs": [int(x) for x in cdf]}
        )
    return {
        "note": "logprobs are a float64 log-softmax. Rounding is numpy round (half to even), then max(1, x), then cumulative sum.",
        "scale_factor": llama_zip.FREQ_SCALE_FACTOR,
        "cases": cases,
    }


def window_vectors():
    cases = []

    def case(arg, n_ctx):
        # Mirrors main(): returns the resulting overlap or the error kind.
        try:
            if arg.endswith("%"):
                percent = float(arg[:-1])
                if not (0 <= percent <= 100):
                    return {"arg": arg, "n_ctx": n_ctx, "error": "range_percent"}
                value = int(percent / 100 * (n_ctx - 1))
            else:
                value = int(arg)
                if value < 0:
                    value += n_ctx
                if not (0 <= value < n_ctx):
                    return {"arg": arg, "n_ctx": n_ctx, "error": "range_tokens"}
        except ValueError:
            return {"arg": arg, "n_ctx": n_ctx, "error": "not_integer"}
        return {"arg": arg, "n_ctx": n_ctx, "overlap": value}

    for n_ctx in (64, 100, 1000):
        for arg in ["0%", "0", "50%", "100%", "25%", "33.3%", "99.9%", "12.5%", "0.5%",
                    "1", "10", str(n_ctx - 1), "-1", "-10", str(-n_ctx), " 7 ", "+3", "1_0",
                    "101%", "-1%", "150%", "nan%", "inf%", "-inf%",
                    str(n_ctx), str(n_ctx + 1), str(-n_ctx - 1),
                    "abc", "", "%", "1.5", "1e2", "0x10", "5 %", "ten%", "50"]:
            cases.append(case(arg, n_ctx))
    return {
        "note": "overlap is computed by main() from --window-overlap and the model's n_ctx; error kinds map to the parser.error messages",
        "messages": {
            "range_percent": "window overlap must be in the range [0%, 100%]",
            "range_tokens": "window overlap must be in the range [-{n_ctx}, {n_ctx_minus_1}]",
            "not_integer": "window overlap must be an integer (number of tokens) or a percentage (of the model's context length)",
        },
        "cases": cases,
    }


def llamazip_vectors():
    model = {
        "bos": fake_llama.BOS,
        "eos": fake_llama.EOS,
        "byte_base": fake_llama.BYTE_BASE,
        "merges_hex": [m.hex() for m in fake_llama.MERGES],
        "vocab_size": fake_llama.VOCAB_SIZE,
        "tokenizer": "greedy longest match over merges, else one token per byte (id = byte_base + byte). Merge k has id byte_base + 256 + k. With space_prefix, a non empty text is tokenized as b' ' + text. detokenize concatenates the pieces, bos and eos are empty.",
        "logits": "context is the token list given to the model (bos first). h = last*131 + before*31 + len(context)*7, where before is 0 for a one token context. logits[i] = ((h*(i+3) + i*i) % 31) / 2, as float32.",
        "logprobs": "float64 log-softmax",
        "n_ctx_default": fake_llama.DEFAULT_N_CTX,
    }
    texts = [
        ("empty", b""),
        ("one char", b"x"),
        ("the thing", b"the thing"),
        ("sentence", b"the king the thing, the thing the king, in the thing."),
        ("leading space", b" the king"),
        ("two leading spaces", b"  x"),
        ("non ASCII", "café € \U0001f600".encode()),
        ("invalid bytes", b"ab\xff\xfe\xc3(\x80z"),
        ("PUA collision", b"a\xee\x80\x80\xee\x83\xbf\xee\x84\x80b"),
        ("repetitive", b"the " * 30),
        ("all bytes", bytes(range(256))),
    ]
    cases = []
    from fake_llama import FakeLlama

    for space_prefix in (False, True):
        for n_ctx, overlap in [(64, 0), (16, 0), (16, 4), (16, 15), (8, 3)]:
            for name, data in texts:
                FakeLlama.space_prefix = space_prefix
                z = llama_zip.LlamaZip("fake", n_ctx=n_ctx)
                compressed = bytes(z.compress(data, overlap))
                assert bytes(z.decompress(compressed, overlap)) == data
                cases.append(
                    {
                        "name": name,
                        "space_prefix": space_prefix,
                        "n_ctx": n_ctx,
                        "window_overlap": overlap,
                        "input_hex": data.hex(),
                        "compressed_hex": compressed.hex(),
                    }
                )
    FakeLlama.space_prefix = False
    return {"model": model, "cases": cases}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("arithmetic.json", arithmetic_vectors())
    write("utf8.json", utf8_vectors())
    write("base64.json", base64_vectors())
    write("compute_cdf.json", cdf_vectors())
    write("window_overlap.json", window_vectors())
    write("llamazip_fake_model.json", llamazip_vectors())


if __name__ == "__main__":
    main()
