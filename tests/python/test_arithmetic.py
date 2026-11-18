import random

import numpy as np
import pytest

import llama_zip
from llama_zip import ArithmeticCoderBase, Decoder, Encoder
from vectors import case_ids, load_vectors

VECTORS = load_vectors("arithmetic.json")


def encode(cdfs, steps):
    encoder = Encoder()
    for cdf_index, symbol in steps:
        encoder.encode_symbol(cdfs[cdf_index], symbol)
    encoder.finish()
    return bytes(encoder.get_encoded())


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_encode_matches_vector(case):
    assert encode(case["cdfs"], case["steps"]).hex() == case["encoded_hex"]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_decode_matches_vector(case):
    decoder = Decoder(bytes.fromhex(case["encoded_hex"]))
    decoded = [int(decoder.decode_symbol(case["cdfs"][i])) for i, _ in case["steps"]]
    assert decoded == [s for _, s in case["steps"]]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_decode_accepts_numpy_cdf(case):
    decoder = Decoder(bytes.fromhex(case["encoded_hex"]))
    cdfs = [np.array(c, dtype=np.float64) for c in case["cdfs"]]
    decoded = [int(decoder.decode_symbol(cdfs[i])) for i, _ in case["steps"]]
    assert decoded == [s for _, s in case["steps"]]


@pytest.mark.parametrize(
    "case", VECTORS["decode_only"], ids=case_ids(VECTORS["decode_only"])
)
def test_decode_arbitrary_data(case):
    decoder = Decoder(bytes.fromhex(case["data_hex"]))
    symbols = [int(decoder.decode_symbol(case["cdf"])) for _ in case["symbols"]]
    assert symbols == case["symbols"]


def test_underflow_vectors_exercise_underflow():
    by_name = {c["name"]: c for c in VECTORS["cases"]}
    assert by_name["underflow middle symbol repeated"]["max_underflow"] == 47
    assert sum(c["max_underflow"] > 0 for c in VECTORS["cases"]) >= 8


def test_empty_message_is_one_set_bit():
    assert encode([[1, 2]], []) == b"\x80"


def test_single_symbol_alphabet_costs_one_bit_whatever_the_length():
    assert encode([[1]], [[0, 0]] * 1000) == b"\x80"


def test_binary_alphabet_hand_computed():
    # 0 -> low half (bit 0), 1 -> high half (bit 1); finish() adds a trailing 1.
    assert encode([[1, 2]], [[0, 0]]) == b"\x40"
    assert encode([[1, 2]], [[0, 1]]) == b"\xc0"
    assert encode([[1, 2]], [[0, 0], [0, 1], [0, 0], [0, 1]]) == bytes([0b01011000])


def test_uniform_256_writes_the_symbols_as_bytes():
    cdf = list(range(1, 257))
    data = encode([cdf], [[0, b] for b in range(256)])
    assert data[:256] == bytes(range(256))


def test_initial_state():
    coder = ArithmeticCoderBase()
    assert coder.low == 0
    assert coder.high == (1 << 64) - 1
    assert coder.half_range == 1 << 63
    assert coder.quarter_range == 1 << 62
    assert coder.state_mask == (1 << 64) - 1


def test_base_class_hooks_are_abstract():
    coder = ArithmeticCoderBase()
    with pytest.raises(NotImplementedError):
        coder.shift()
    with pytest.raises(NotImplementedError):
        coder.underflow()


def test_state_stays_in_64_bits():
    rng = random.Random(7)
    encoder = Encoder()
    cdf = [1, 5, 1 << 32, (1 << 32) + 3]
    for _ in range(500):
        encoder.encode_symbol(cdf, rng.randrange(4))
        assert 0 <= encoder.low <= encoder.high <= (1 << 64) - 1
        assert (encoder.high - encoder.low) >= encoder.quarter_range


def test_encoder_underflow_counter_is_flushed_on_shift():
    encoder = Encoder()
    for _ in range(5):
        encoder.encode_symbol([1, 2, 3], 1)
    assert encoder.num_underflow >= 5
    pending = encoder.num_underflow
    assert encoder.get_encoded() == bytearray()
    encoder.encode_symbol([1, 2, 3], 0)
    assert encoder.num_underflow == 0
    assert len(encoder.get_encoded()) * 8 >= pending


def test_encoder_bit_packing():
    encoder = Encoder()
    for bit in (1, 0, 1, 1, 0, 0, 0, 1, 1):
        encoder.append_bit(bit)
    assert bytes(encoder.get_encoded()) == bytes([0b10110001, 0b10000000])


def test_decoder_reads_zero_bits_past_the_end():
    decoder = Decoder(b"")
    assert decoder.code == 0
    assert decoder.read_code_bit() == 0
    decoder = Decoder(b"\xff")
    assert decoder.code == 0xFF << 56


def test_decoder_reads_msb_first():
    decoder = Decoder(b"\xa5" + b"\x00" * 7 + b"\x80")
    assert decoder.code == 0xA5 << 56
    assert decoder.read_code_bit() == 1
    assert decoder.read_code_bit() == 0


def test_trailing_zero_bytes_do_not_change_the_decoded_symbols():
    case = VECTORS["cases"][-1]
    data = bytes.fromhex(case["encoded_hex"]) + b"\x00" * 9
    decoder = Decoder(data)
    decoded = [int(decoder.decode_symbol(case["cdfs"][i])) for i, _ in case["steps"]]
    assert decoded == [s for _, s in case["steps"]]


def test_random_round_trips():
    rng = random.Random(2024)
    for _ in range(200):
        size = rng.randint(1, 40)
        total = rng.choice([size, size * 3, 1000, 1 << 16, 1 << 32, (1 << 32) + size])
        total = max(total, size)
        cuts = sorted(rng.sample(range(1, total), size - 1)) if size > 1 else []
        cdf = cuts + [total]
        steps = [[0, rng.randrange(size)] for _ in range(rng.randint(0, 150))]
        data = encode([cdf], steps)
        decoder = Decoder(data)
        assert [int(decoder.decode_symbol(cdf)) for _ in steps] == [s for _, s in steps]


def test_compressed_size_tracks_information_content():
    # 4096 symbols at 8 bits each fill 4096 bytes, and finish() starts one more.
    data = encode([list(range(1, 257))], [[0, i % 256] for i in range(4096)])
    assert len(data) == 4097


def test_module_constants():
    assert llama_zip.NUM_STATE_BITS == 64
    assert llama_zip.FREQ_SCALE_FACTOR == 1 << 32
