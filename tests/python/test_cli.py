import base64
import io
import runpy
import sys

import pytest

import llama_zip
from fake_llama import FakeLlama
from llama_zip import LlamaZip
from vectors import load_vectors

WINDOW = load_vectors("window_overlap.json")


@pytest.fixture
def run(monkeypatch, capsysbinary):
    def run(*argv, stdin=b"", inputs=None):
        monkeypatch.setattr(sys, "argv", ["llama_zip.py", "fake.gguf", *argv])
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin)))
        if inputs is not None:
            queue = iter(inputs)

            def fake_input(prompt=""):
                sys.stderr.write("[prompt]")
                item = next(queue)
                if isinstance(item, BaseException):
                    raise item
                return item

            monkeypatch.setattr("builtins.input", fake_input)
        code = None
        try:
            llama_zip.main()
        except SystemExit as exit_:
            code = exit_.code
        captured = capsysbinary.readouterr()
        return code, captured.out, captured.err.decode("utf-8", "replace")

    return run


def reference(data, overlap=0, n_ctx=64):
    return bytes(LlamaZip("fake.gguf", n_ctx=n_ctx).compress(data, overlap))


class TestArgumentErrors:
    def test_mode_is_required(self, run):
        code, out, err = run()
        assert code == 2
        assert "one of the arguments -c/--compress -d/--decompress -i/--interactive is required" in err

    def test_modes_are_mutually_exclusive(self, run):
        code, _, err = run("-c", "x", "-d", "y")
        assert code == 2
        assert "not allowed with argument" in err

    def test_compress_and_interactive_are_exclusive(self, run):
        assert run("-c", "-i")[0] == 2

    def test_unknown_format_is_rejected(self, run):
        code, _, err = run("-c", "x", "-f", "hex")
        assert code == 2
        assert "invalid choice" in err

    def test_interactive_rejects_binary(self, run):
        code, _, err = run("-i", "-f", "binary")
        assert code == 2
        assert "interactive mode only supports base64 compressed data" in err

    def test_missing_model_path(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["llama_zip.py", "-c", "x"])
        with pytest.raises(SystemExit) as info:
            llama_zip.main()
        assert info.value.code == 2

    def test_n_ctx_must_be_an_integer(self, run):
        code, _, err = run("-c", "x", "--n-ctx", "big")
        assert code == 2
        assert "invalid int value" in err

    def test_help_exits_cleanly(self, run):
        code, out, _ = run("-h")
        assert code == 0
        assert b"LLM-powered lossless compression tool" in out
        assert b"default: 0%" in out

    def test_arguments_error_before_the_model_loads(self, run):
        run("-i", "-f", "binary")
        assert FakeLlama.instances == []


class TestModelOptions:
    def test_defaults_are_forwarded(self, run):
        run("-c", "x")
        (model,) = FakeLlama.instances
        assert model.model_path == "fake.gguf"
        assert (model.n_ctx(), model.n_gpu_layers, model.use_mlock) == (64, -1, False)

    def test_options_are_forwarded(self, run):
        run("-c", "x", "--n-ctx", "48", "--n-gpu-layers", "7", "--use-mlock")
        (model,) = FakeLlama.instances
        assert (model.n_ctx(), model.n_gpu_layers, model.use_mlock) == (48, 7, True)

    def test_loading_message_goes_to_stderr(self, run):
        _, out, err = run("-c", "x")
        assert "Loading model..." in err
        assert b"Loading" not in out


class TestCompress:
    def test_binary_output_from_arguments(self, run):
        code, out, _ = run("-c", "the", "thing")
        assert code is None
        assert out == reference(b"the thing")

    def test_arguments_are_joined_with_spaces(self, run):
        assert run("-c", "a", "b", "c")[1] == reference(b"a b c")

    def test_non_ascii_argument(self, run):
        assert run("-c", "café €")[1] == reference("café €".encode())

    def test_reads_stdin_without_arguments(self, run):
        data = b"from stdin \xff\xfe\n"
        assert run("-c", stdin=data)[1] == reference(data)

    def test_empty_stdin(self, run):
        assert run("-c", stdin=b"")[1] == reference(b"")

    def test_base64_output(self, run):
        _, out, _ = run("-c", "the thing", "-f", "base64")
        assert out == base64.b64encode(reference(b"the thing"))

    def test_long_option_names(self, run):
        _, out, _ = run("--compress", "x", "--compressed-format", "base64")
        assert out == base64.b64encode(reference(b"x"))

    def test_window_overlap_changes_the_output(self, run):
        data = "the thing, the king; " * 8
        out_a = run("-c", data, "--n-ctx", "16", "-w", "0")[1]
        out_b = run("-c", data, "--n-ctx", "16", "-w", "5")[1]
        assert out_a == reference(data.encode(), 0, 16)
        assert out_b == reference(data.encode(), 5, 16)
        assert out_a != out_b

    def test_progress_bar_goes_to_stderr_only(self, run):
        _, out, err = run("-c", "the thing")
        assert "Compressing" not in out.decode("latin-1")

    def test_keyboard_interrupt_is_swallowed(self, run, monkeypatch):
        def boom(self, *_):
            raise KeyboardInterrupt

        monkeypatch.setattr(LlamaZip, "compress", boom)
        code, out, err = run("-c", "x")
        assert code is None and out == b""
        assert err.endswith("\n")


class TestDecompress:
    def test_base64_argument(self, run):
        packed = base64.b64encode(reference(b"the thing")).decode()
        code, out, _ = run("-d", packed, "-f", "base64")
        assert code is None
        assert out == b"the thing"

    def test_base64_argument_is_cleaned_up(self, run):
        packed = base64.b64encode(reference(b"the thing")).decode()
        noisy = packed[:4] + " \n" + packed[4:] + "\n"
        assert run("-d", noisy, "-f", "base64")[1] == b"the thing"

    def test_binary_from_stdin(self, run):
        _, out, _ = run("-d", stdin=reference(b"binary \xff\xfe\x80 data"))
        assert out == b"binary \xff\xfe\x80 data"

    def test_base64_from_stdin(self, run):
        packed = base64.b64encode(reference(b"x y z"))
        assert run("-d", "-f", "base64", stdin=packed + b"\n")[1] == b"x y z"

    def test_only_the_first_argument_is_used(self, run):
        packed = base64.b64encode(reference(b"first")).decode()
        other = base64.b64encode(reference(b"second")).decode()
        assert run("-d", packed, other, "-f", "base64")[1] == b"first"

    def test_window_overlap_round_trip(self, run):
        data = ("the thing, the king; " * 8).encode()
        packed = base64.b64encode(reference(data, 5, 16)).decode()
        out = run("-d", packed, "-f", "base64", "--n-ctx", "16", "-w", "5")[1]
        assert out == data

    def test_decompress_with_percent_overlap(self, run):
        data = ("the thing, the king; " * 8).encode()
        overlap = int(0.5 * (16 - 1))
        packed = base64.b64encode(reference(data, overlap, 16)).decode()
        out = run("-d", packed, "-f", "base64", "--n-ctx", "16", "-w", "50%")[1]
        assert out == data

    def test_keyboard_interrupt_is_swallowed(self, run, monkeypatch):
        def boom(self, *_):
            raise KeyboardInterrupt

        monkeypatch.setattr(LlamaZip, "decompress", boom)
        code, out, err = run("-d", "AAAA", "-f", "base64")
        assert code is None
        assert err.endswith("\n")


class TestWindowOverlapParsing:
    @pytest.fixture
    def seen(self, monkeypatch):
        calls = []

        def spy(self, data, window_overlap=0):
            calls.append(window_overlap)
            return b""

        monkeypatch.setattr(LlamaZip, "compress", spy)
        return calls

    @pytest.mark.parametrize(
        "case",
        WINDOW["cases"],
        ids=[f"ctx{c['n_ctx']}-{c['arg']!r}" for c in WINDOW["cases"]],
    )
    def test_vectors(self, run, seen, case):
        n_ctx = case["n_ctx"]
        code, _, err = run("-c", "x", "--n-ctx", str(n_ctx), f"--window-overlap={case['arg']}")
        if "overlap" in case:
            assert code is None, err
            assert seen == [case["overlap"]]
        else:
            assert code == 2
            assert seen == []
            template = WINDOW["messages"][case["error"]]
            assert template.format(n_ctx=n_ctx, n_ctx_minus_1=n_ctx - 1) in err

    def test_default_overlap_is_zero(self, run, seen):
        run("-c", "x")
        assert seen == [0]

    def test_negative_number_as_a_separate_argument(self, run, seen):
        run("-c", "x", "--n-ctx", "64", "-w", "-1")
        assert seen == [63]

    def test_percentage_uses_n_ctx_minus_one(self, run, seen):
        run("-c", "x", "--n-ctx", "11", "-w", "100%")
        assert seen == [10]

    def test_message_is_a_parser_error(self, run, seen):
        _, _, err = run("-c", "x", "-w", "oops")
        assert "usage:" in err
        assert ": error: window overlap must be an integer" in err


class TestInteractive:
    def prompt_count(self, err):
        return err.count("[prompt]")

    def test_compress_then_exit_on_ctrl_c(self, run):
        code, out, err = run("-i", inputs=["Hi there!", KeyboardInterrupt()])
        assert code is None
        assert out == base64.b64encode(reference(b"Hi there!"))
        assert self.prompt_count(err) == 2

    def test_base64_input_is_decompressed(self, run):
        packed = base64.b64encode(reference(b"Hi there!")).decode()
        _, out, _ = run("-i", inputs=[packed, KeyboardInterrupt()])
        assert out == b"Hi there!"

    def test_compress_decompress_session(self, run):
        packed = base64.b64encode(reference(b"Hi there!")).decode()
        _, out, err = run("-i", inputs=["Hi there!", packed, "Bye now.", KeyboardInterrupt()])
        expected = (
            base64.b64encode(reference(b"Hi there!"))
            + b"Hi there!"
            + base64.b64encode(reference(b"Bye now."))
        )
        assert out == expected
        assert self.prompt_count(err) == 4

    def test_each_turn_ends_with_a_blank_line_on_stderr(self, run):
        _, _, err = run("-i", inputs=["Hi there!", KeyboardInterrupt()])
        assert "\n\n[prompt]" in err

    def test_empty_line_compresses_the_empty_string(self, run):
        _, out, _ = run("-i", inputs=["", KeyboardInterrupt()])
        assert out == base64.b64encode(reference(b""))

    def test_plain_word_made_of_base64_characters_is_treated_as_base64(self, run, monkeypatch):
        # "hello" only uses base64 characters, so interactive mode decodes it.
        calls = []
        monkeypatch.setattr(
            LlamaZip, "decompress", lambda self, data, overlap=0: calls.append(bytes(data))
        )
        run("-i", inputs=["hello", KeyboardInterrupt()])
        assert calls == [llama_zip.robust_b64decode(b"hello")]

    def test_equals_only_input_is_routed_to_decompress(self, run, monkeypatch):
        calls = []
        monkeypatch.setattr(
            LlamaZip, "decompress", lambda self, data, overlap=0: calls.append(bytes(data))
        )
        run("-i", inputs=["===", KeyboardInterrupt()])
        assert calls == [b""]

    def test_non_base64_character_makes_it_a_compression_request(self, run, monkeypatch):
        calls = []
        monkeypatch.setattr(
            LlamaZip, "compress", lambda self, data, overlap=0: calls.append(bytes(data)) or b""
        )
        run("-i", inputs=["abc-def", "AAAA!", KeyboardInterrupt()])
        assert calls == [b"abc-def", b"AAAA!"]

    def test_unicode_error_is_reported_and_the_loop_continues(self, run):
        bad = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        _, out, err = run("-i", inputs=[bad, "Hi there!", KeyboardInterrupt()])
        assert "error: interactive mode only supports UTF-8 input" in err
        assert out == base64.b64encode(reference(b"Hi there!"))

    def test_interrupting_a_decompression_returns_to_the_prompt(self, run, monkeypatch):
        calls = []

        def interrupted(self, data, overlap=0):
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return b""

        monkeypatch.setattr(LlamaZip, "decompress", interrupted)
        code, _, err = run("-i", inputs=["AAAA", "BBBB", KeyboardInterrupt()])
        assert code is None
        assert len(calls) == 2
        assert self.prompt_count(err) == 3

    def test_window_overlap_applies_in_interactive_mode(self, run, monkeypatch):
        seen = []
        monkeypatch.setattr(
            LlamaZip, "compress", lambda self, d, overlap=0: seen.append(overlap) or b""
        )
        run("-i", "--n-ctx", "20", "-w", "50%", inputs=["Hi there!", KeyboardInterrupt()])
        assert seen == [9]

    def test_explicit_base64_format_is_accepted(self, run):
        code, out, _ = run("-i", "-f", "base64", inputs=["Hi there!", KeyboardInterrupt()])
        assert code is None and out


def test_no_mode_selected_does_nothing(monkeypatch):
    # argparse makes the mode group required, so this branch needs a stub parser.
    import argparse

    class Parser:
        def parse_args(self):
            return argparse.Namespace(
                compressed_format="binary",
                interactive=False,
                model_path="fake.gguf",
                use_mlock=False,
                n_ctx=16,
                n_gpu_layers=-1,
                overlap="0",
                string=None,
                compressed=None,
            )

    monkeypatch.setattr(llama_zip, "make_arg_parser", Parser)
    assert llama_zip.main() is None


def test_running_as_a_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["llama_zip.py", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(llama_zip.__file__, run_name="__main__")
    assert info.value.code == 0
    assert "LLM-powered lossless compression tool" in capsys.readouterr().out
