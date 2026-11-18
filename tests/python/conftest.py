import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llama import FakeLlama  # noqa: E402

# Detect the real package before the stub hides it (used by the integration test).
REAL_LLAMA_CPP_AVAILABLE = (
    "llama_cpp" not in sys.modules and importlib.util.find_spec("llama_cpp") is not None
)

stub = types.ModuleType("llama_cpp")
stub.Llama = FakeLlama
sys.modules["llama_cpp"] = stub

import llama_zip  # noqa: E402,F401


@pytest.fixture(autouse=True)
def reset_fake_llama():
    FakeLlama.instances = []
    FakeLlama.space_prefix = False
    FakeLlama.on_logits = None
    yield
    FakeLlama.on_logits = None
    FakeLlama.space_prefix = False
