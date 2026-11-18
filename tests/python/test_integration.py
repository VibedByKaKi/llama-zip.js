"""Optional end to end test with the real llama-cpp-python and a small GGUF model.

Skipped when llama_cpp cannot be imported or the model file is missing. Set
LLAMA_ZIP_TEST_MODEL to use another model.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from conftest import REAL_LLAMA_CPP_AVAILABLE, ROOT

DEFAULT_MODEL = (
    "/tmp/claude-1000/-home-kaki-Documents-git-github-com-VibedByKaKi-llmzip-desktop/"
    "4eff2479-8a67-41fd-9301-2500fcb1ab44/scratchpad/models/SmolLM2-135M.Q8_0.gguf"
)
MODEL = os.environ.get("LLAMA_ZIP_TEST_MODEL", DEFAULT_MODEL)

pytestmark = [
    pytest.mark.skipif(
        not REAL_LLAMA_CPP_AVAILABLE, reason="llama_cpp is not importable"
    ),
    pytest.mark.skipif(
        not Path(MODEL).is_file(), reason=f"model file not found: {MODEL}"
    ),
]


@pytest.fixture(scope="module")
def real_module():
    # The other tests run against a stub, so load a separate copy of the source
    # with the real llama_cpp for this module only.
    stub = sys.modules["llama_cpp"]
    del sys.modules["llama_cpp"]
    try:
        import llama_cpp  # noqa: F401

        spec = importlib.util.spec_from_file_location(
            "llama_zip_real", ROOT / "llama_zip.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.modules["llama_cpp"] = stub
    return module


@pytest.fixture(scope="module")
def zipper(real_module):
    return real_module.LlamaZip(MODEL, n_ctx=128, n_gpu_layers=0)


TEXT = b"The quick brown fox jumps over the lazy dog. " * 3


def test_round_trip_compresses_english(zipper):
    compressed = zipper.compress(TEXT)
    assert len(compressed) < len(TEXT)
    assert bytes(zipper.decompress(compressed)) == TEXT


def test_round_trip_with_window_overlap(zipper):
    data = TEXT * 4  # longer than the 128 token window
    compressed = zipper.compress(data, window_overlap=32)
    assert bytes(zipper.decompress(compressed, window_overlap=32)) == data


@pytest.mark.parametrize(
    "data",
    [b"", b" leading space", "café € \U0001f600".encode(), b"\xff\xfe\x80 raw"],
)
def test_round_trip_edge_inputs(zipper, data):
    assert bytes(zipper.decompress(zipper.compress(data))) == data
