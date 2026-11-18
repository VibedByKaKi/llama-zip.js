import numpy as np
import pytest

from fake_llama import FakeLlama, fake_logits
from llama_zip import FREQ_SCALE_FACTOR, LlamaZip
from vectors import case_ids, load_vectors

VECTORS = load_vectors("compute_cdf.json")


@pytest.fixture
def zipper():
    return LlamaZip("fake", n_ctx=32)


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_compute_cdf_vectors(zipper, case):
    cdf = zipper.compute_cdf(np.array(case["logits"], dtype=np.float32))
    assert [int(x) for x in cdf] == case["cum_freqs"]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_cdf_is_strictly_increasing_with_floor_of_one(zipper, case):
    cdf = zipper.compute_cdf(np.array(case["logits"], dtype=np.float32))
    freqs = np.diff(np.concatenate([[0], cdf]))
    assert (freqs >= 1).all()
    assert len(cdf) == len(case["logits"])


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_cdf_total_is_scale_factor_plus_floor_bumps(zipper, case):
    cdf = zipper.compute_cdf(np.array(case["logits"], dtype=np.float32))
    n = len(cdf)
    assert FREQ_SCALE_FACTOR - n <= cdf[-1] <= FREQ_SCALE_FACTOR + n


def test_scale_factor_constant_in_vectors():
    assert VECTORS["scale_factor"] == FREQ_SCALE_FACTOR == 1 << 32


def test_uniform_logits_split_evenly(zipper):
    cdf = zipper.compute_cdf(np.zeros(8, dtype=np.float32))
    assert [int(x) for x in cdf] == [(i + 1) * (1 << 29) for i in range(8)]


def test_zero_probability_tokens_get_frequency_one(zipper):
    cdf = zipper.compute_cdf(np.array([0.0, -200.0, -200.0], dtype=np.float32))
    assert [int(x) for x in np.diff(np.concatenate([[0], cdf]))][1:] == [1, 1]


def test_cdf_is_invariant_to_a_constant_shift(zipper):
    a = zipper.compute_cdf(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = zipper.compute_cdf(np.array([51.0, 52.0, 53.0], dtype=np.float32))
    assert (a == b).all()


def test_cdf_is_float64(zipper):
    assert zipper.compute_cdf(fake_logits([0, 4])).dtype == np.float64


def test_cdf_tracks_the_logits_order(zipper):
    cdf = zipper.compute_cdf(np.array([0.0, 3.0, 1.0], dtype=np.float32))
    freqs = np.diff(np.concatenate([[0], cdf]))
    assert freqs[1] > freqs[2] > freqs[0]


def test_cdf_for_full_fake_vocab_is_monotonic(zipper):
    for context in ([0], [0, 5], [0, 5, 9, 12]):
        cdf = zipper.compute_cdf(fake_logits(context))
        assert (np.diff(cdf) >= 1).all()


def test_compute_cdf_calls_model_logits_to_logprobs(zipper, monkeypatch):
    seen = []
    original = FakeLlama.logits_to_logprobs

    def spy(self, logits):
        seen.append(logits)
        return original(self, logits)

    monkeypatch.setattr(FakeLlama, "logits_to_logprobs", spy)
    zipper.compute_cdf(np.zeros(3, dtype=np.float32))
    assert len(seen) == 1
