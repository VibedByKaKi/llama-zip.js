import signal

import numpy as np
import pytest

import llama_zip
from fake_llama import BOS, EOS, FakeLlama, VOCAB_SIZE
from llama_zip import LlamaZip
from vectors import load_vectors

VECTORS = load_vectors("llamazip_fake_model.json")


def case_id(case):
    prefix = "space" if case["space_prefix"] else "nospace"
    return f"{prefix}-ctx{case['n_ctx']}-ov{case['window_overlap']}-{case['name']}"


CASES = VECTORS["cases"]
IDS = [case_id(c) for c in CASES]


@pytest.fixture
def zipper():
    return LlamaZip("fake.gguf", n_ctx=32)


def make(n_ctx, space_prefix=False):
    FakeLlama.space_prefix = space_prefix
    return LlamaZip("fake.gguf", n_ctx=n_ctx)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_compress_matches_vector(case):
    z = make(case["n_ctx"], case["space_prefix"])
    data = bytes.fromhex(case["input_hex"])
    assert z.compress(data, case["window_overlap"]).hex() == case["compressed_hex"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_decompress_matches_vector(case):
    z = make(case["n_ctx"], case["space_prefix"])
    compressed = bytes.fromhex(case["compressed_hex"])
    assert bytes(z.decompress(compressed, case["window_overlap"])) == bytes.fromhex(
        case["input_hex"]
    )


def test_fake_model_description_matches_the_fake():
    model = VECTORS["model"]
    assert model["vocab_size"] == VOCAB_SIZE
    assert (model["bos"], model["eos"]) == (BOS, EOS)


class TestLoadModel:
    def test_passes_arguments_to_llama(self):
        LlamaZip("some/model.gguf", n_ctx=100, n_gpu_layers=3, use_mlock=True)
        (model,) = FakeLlama.instances
        assert model.model_path == "some/model.gguf"
        assert model.n_ctx() == 100
        assert model.n_gpu_layers == 3
        assert model.use_mlock is True
        assert model.verbose is False

    def test_defaults(self):
        LlamaZip("m.gguf")
        (model,) = FakeLlama.instances
        assert model.n_gpu_layers == -1
        assert model.use_mlock is False
        assert model.n_ctx() == 64

    def test_quiet_by_default(self, capsys):
        LlamaZip("m.gguf")
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""

    def test_verbose_shows_and_clears_loading_message(self, capsys):
        z = LlamaZip("m.gguf", verbose=True)
        assert z.verbose is True
        err = capsys.readouterr().err
        assert err == "Loading model..." + "\r" + " " * len("Loading model...") + "\r"


class TestRoundTrip:
    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"a",
            b"the thing",
            b"  leading spaces",
            "café € \U0001f600".encode(),
            bytes(range(256)),
            b"\xff\xfe\x80",
            b"the " * 100,
        ],
    )
    @pytest.mark.parametrize("overlap", [0, 1, 7, 31])
    def test_round_trip(self, zipper, data, overlap):
        compressed = zipper.compress(data, overlap)
        assert bytes(zipper.decompress(compressed, overlap)) == data

    def test_returns_bytes_like_types(self, zipper):
        assert isinstance(zipper.compress(b"abc"), bytearray)
        assert isinstance(zipper.decompress(zipper.compress(b"abc")), bytearray)

    def test_default_window_overlap_is_zero(self, zipper):
        data = b"the thing " * 10
        assert zipper.compress(data) == zipper.compress(data, 0)
        assert zipper.decompress(zipper.compress(data)) == zipper.decompress(
            zipper.compress(data, 0), 0
        )

    def test_compression_is_deterministic(self, zipper):
        data = b"the king the thing"
        assert zipper.compress(data) == zipper.compress(data)

    def test_empty_input_is_just_the_eos_symbol(self, zipper):
        compressed = zipper.compress(b"")
        assert bytes(zipper.decompress(compressed)) == b""
        assert len(compressed) <= 5

    def test_model_is_reset_each_time(self, zipper):
        model = FakeLlama.instances[0]
        zipper.compress(b"ab")
        zipper.decompress(zipper.compress(b"ab"))
        assert model.reset_calls == 3

    def test_overlap_changes_output_once_the_window_fills(self, zipper):
        data = b"the thing, the king; " * 6
        a = zipper.compress(data, 0)
        b = zipper.compress(data, 10)
        assert a != b
        assert bytes(zipper.decompress(a, 0)) == data
        assert bytes(zipper.decompress(b, 10)) == data

    def test_overlap_is_irrelevant_while_the_window_has_room(self, zipper):
        assert zipper.compress(b"ab", 0) == zipper.compress(b"ab", 20)


class SymbolSpy:
    def __init__(self, monkeypatch):
        self.symbols = []
        original = llama_zip.Encoder.encode_symbol

        def encode_symbol(encoder, cum_freqs, symbol):
            self.symbols.append(int(symbol))
            return original(encoder, cum_freqs, symbol)

        monkeypatch.setattr(llama_zip.Encoder, "encode_symbol", encode_symbol)


class TestCompress:
    def test_encodes_token_ids_followed_by_eos(self, zipper, monkeypatch):
        spy = SymbolSpy(monkeypatch)
        zipper.compress(b"the x")
        expected = FakeLlama.instances[0].tokenize(b"the x", add_bos=False) + [EOS]
        assert spy.symbols == expected

    def test_empty_input_encodes_only_eos(self, zipper, monkeypatch):
        spy = SymbolSpy(monkeypatch)
        zipper.compress(b"")
        assert spy.symbols == [EOS]

    def test_bytes_are_mapped_to_utf8_before_tokenizing(self, zipper, monkeypatch):
        spy = SymbolSpy(monkeypatch)
        zipper.compress(b"\xff")
        pua = chr(llama_zip.PUA_START + 0xFF).encode()
        expected = FakeLlama.instances[0].tokenize(pua, add_bos=False) + [EOS]
        assert spy.symbols == expected
        assert len(expected) == 4  # three single byte tokens plus EOS

    def test_context_starts_with_bos_and_grows(self, zipper):
        zipper.compress(b"abc")
        (call,) = FakeLlama.instances[0].generate_calls
        assert call == [BOS]

    def test_window_restarts_with_overlap_tokens(self):
        z = make(8)
        data = b"abcdefghijklmnopqrstuvwxyz"
        tokens = z.model.tokenize(data, add_bos=False)
        z.compress(data, 3)
        calls = z.model.generate_calls
        assert calls[0] == [BOS]
        assert len(calls) > 3
        consumed = 0
        for call in calls:
            # A window of 8 holds BOS, up to 3 overlap tokens and the new ones.
            start = max(0, consumed - 3)
            assert call == [BOS] + tokens[start:consumed]
            # Sampling continues until the context holds n_ctx tokens, inclusive.
            consumed += 8 - len(call) + 1

    def test_window_with_zero_overlap_starts_from_bos_only(self):
        z = make(6)
        z.compress(b"abcdefghijklmn", 0)
        assert all(call == [BOS] for call in z.model.generate_calls)
        assert len(z.model.generate_calls) >= 3

    def test_context_never_exceeds_n_ctx(self):
        z = make(8)
        seen = []
        FakeLlama.on_logits = lambda ctx: seen.append(len(ctx))
        z.compress(b"abcdefghijklmnopqrstuvwxyz", 5)
        assert max(seen) == 8

    def test_maximum_overlap_still_makes_progress(self):
        z = make(8)
        data = b"abcdefghijkl"
        compressed = z.compress(data, 7)
        assert bytes(z.decompress(compressed, 7)) == data

    def test_progress_bar_and_silence(self, capfd):
        z = LlamaZip("m.gguf", n_ctx=32)
        z.compress(b"the thing")
        out = capfd.readouterr()
        assert out.out == "" and out.err == ""

    def test_verbose_compress_uses_a_progress_bar(self, capfd):
        z = LlamaZip("m.gguf", n_ctx=32, verbose=True)
        z.compress(b"the thing")
        err = capfd.readouterr().err
        assert "Compressing" in err


class TestDecompress:
    def test_stops_at_eos_and_excludes_it(self, zipper):
        compressed = zipper.compress(b"ab")
        assert bytes(zipper.decompress(compressed)) == b"ab"

    def test_eos_only_stream(self, zipper):
        assert bytes(zipper.decompress(zipper.compress(b""))) == b""

    def test_trailing_garbage_after_the_stream_is_ignored(self, zipper):
        compressed = zipper.compress(b"the king")
        assert bytes(zipper.decompress(bytes(compressed) + b"\x00" * 8)) == b"the king"

    def test_verbose_streams_output_to_stdout(self, capsysbinary):
        z = LlamaZip("m.gguf", n_ctx=32, verbose=True)
        compressed = z.compress(b"the thing")
        capsysbinary.readouterr()
        result = z.decompress(compressed)
        assert capsysbinary.readouterr().out == b"the thing" == bytes(result)

    def test_non_verbose_does_not_stream(self, capsysbinary, zipper):
        zipper.decompress(zipper.compress(b"the thing"))
        assert capsysbinary.readouterr().out == b""

    def test_multibyte_character_split_across_tokens(self, zipper):
        data = "€\U0001f600".encode()  # single byte tokens split each character
        assert bytes(zipper.decompress(zipper.compress(data))) == data

    def test_invalid_bytes_survive(self, zipper):
        data = b"a\xc3\xffb\x80"
        assert bytes(zipper.decompress(zipper.compress(data))) == data

    def test_pua_collisions_survive(self, zipper):
        data = "ab".encode()
        assert bytes(zipper.decompress(zipper.compress(data))) == data

    def test_window_restarts_use_overlap_tokens(self):
        z = make(8)
        data = b"abcdefghijklmnopqrstuvwxyz"
        compressed = z.compress(data, 3)
        z.model.generate_calls.clear()
        assert bytes(z.decompress(compressed, 3)) == data
        calls = z.model.generate_calls
        assert calls[0] == [BOS]
        assert all(c[0] == BOS for c in calls)
        assert max(len(c) for c in calls[1:]) == 1 + 3

    def test_mismatched_overlap_gives_a_different_stream(self, zipper):
        data = b"the thing, the king; " * 6
        assert zipper.compress(data, 4) != zipper.compress(data, 0)


class TestSpacePrefix:
    def test_detects_model_that_adds_a_prefix(self):
        assert make(32, True).tokenizer_adds_space_prefix() is True

    def test_detects_model_without_a_prefix(self):
        assert make(32, False).tokenizer_adds_space_prefix() is False

    def test_first_token_space_is_stripped_when_the_tokenizer_adds_one(self):
        z = make(32, True)
        assert z.model.detokenize(z.model.tokenize(b"x", add_bos=False)) == b" x"
        compressed = z.compress(b"x")
        assert bytes(z.decompress(compressed)) == b"x"

    def test_leading_space_in_the_input_is_kept(self):
        z = make(32, True)
        for data in (b" x", b"  x", b" the thing"):
            assert bytes(z.decompress(z.compress(data))) == data

    def test_only_the_first_token_is_stripped(self):
        z = make(32, True)
        data = b"the  thing   x"
        assert bytes(z.decompress(z.compress(data))) == data

    def test_without_the_special_case_the_prefix_would_leak(self):
        prefixed = make(32, True)
        compressed = prefixed.compress(b"x")
        plain = make(32, False)
        # A model that does not add the prefix keeps the space it decoded.
        assert bytes(plain.decompress(compressed)) == b" x"

    def test_stripping_applies_to_the_first_token_of_each_decompress(self):
        z = make(32, True)
        compressed = z.compress(b"the thing")
        assert bytes(z.decompress(compressed)) == b"the thing"
        assert bytes(z.decompress(compressed)) == b"the thing"

    def test_window_restart_does_not_strip_again(self):
        z = make(8, True)
        data = b"a b c d e f g h i j k l m n o p"
        assert bytes(z.decompress(z.compress(data, 2), 2)) == data

    def test_probe_runs_once_and_only_for_a_first_token_with_a_space(self, monkeypatch):
        calls = []
        original = LlamaZip.tokenizer_adds_space_prefix
        monkeypatch.setattr(
            LlamaZip,
            "tokenizer_adds_space_prefix",
            lambda self: calls.append(1) or original(self),
        )
        plain = make(32, False)
        plain.decompress(plain.compress(b"abc"))
        assert calls == []  # the first token "a" has no space to strip
        prefixed = make(32, True)
        prefixed.decompress(prefixed.compress(b"abc"))
        assert calls == [1]  # only the first token is probed


class TestSigint:
    def test_interrupt_truncates_the_stream_but_stays_decodable(self):
        z = make(64)
        data = b"abcdefghijklmnop"
        count = 0

        def hook(_):
            nonlocal count
            count += 1
            if count == 4:
                signal.raise_signal(signal.SIGINT)

        FakeLlama.on_logits = hook
        compressed = z.compress(data)
        FakeLlama.on_logits = None
        assert bytes(z.decompress(compressed)) == data[:3]

    def test_handler_is_restored(self):
        z = make(64)
        before = signal.getsignal(signal.SIGINT)
        z.compress(b"abc")
        assert signal.getsignal(signal.SIGINT) is before

    def test_default_handler_is_not_triggered(self):
        z = make(64)
        FakeLlama.on_logits = lambda _: signal.raise_signal(signal.SIGINT)
        compressed = z.compress(b"abcdef")  # no KeyboardInterrupt escapes
        FakeLlama.on_logits = None
        assert bytes(z.decompress(compressed)) == b""

    def test_interrupt_on_the_last_token_changes_nothing(self):
        z = make(64)
        data = b"abc"
        expected = z.compress(data)
        count = 0

        def hook(_):
            nonlocal count
            count += 1
            if count == 4:  # the EOS step
                signal.raise_signal(signal.SIGINT)

        FakeLlama.on_logits = hook
        assert z.compress(data) == expected

    def test_verbose_interrupt_prints_a_newline(self, capfd):
        z = LlamaZip("m.gguf", n_ctx=64, verbose=True)
        count = 0

        def hook(_):
            nonlocal count
            count += 1
            if count == 2:
                signal.raise_signal(signal.SIGINT)

        FakeLlama.on_logits = hook
        z.compress(b"abcdef")
        assert "\n" in capfd.readouterr().err

    def test_interrupt_across_windows(self):
        z = make(6)
        count = 0

        def hook(_):
            nonlocal count
            count += 1
            if count == 9:
                signal.raise_signal(signal.SIGINT)

        FakeLlama.on_logits = hook
        compressed = z.compress(b"abcdefghijklmnopqrst", 2)
        FakeLlama.on_logits = None
        assert bytes(z.decompress(compressed, 2)) == b"abcdefgh"


def test_vocab_logits_helper_shapes():
    assert VOCAB_SIZE == VECTORS["model"]["vocab_size"]
    assert np.asarray(FakeLlama("x").logits_to_logprobs([0.0, 0.0])).tolist() == pytest.approx(
        [-np.log(2)] * 2
    )
