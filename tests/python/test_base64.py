import base64

import pytest

import llama_zip
from llama_zip import BASE64, BASE64_EQ, robust_b64decode
from vectors import case_ids, load_vectors

VECTORS = load_vectors("base64.json")


@pytest.mark.parametrize("case", VECTORS["cases"], ids=case_ids(VECTORS["cases"]))
def test_robust_b64decode_vectors(case):
    result = robust_b64decode(bytes.fromhex(case["input_hex"]))
    assert result.hex() == case["output_hex"]


def test_alphabet_constants():
    assert len(BASE64) == 64
    assert len(set(BASE64)) == 64
    assert BASE64_EQ == BASE64 + b"="
    assert BASE64 == base64.b64encode(bytes(range(256)))[:0] + BASE64


def test_padding_is_replaced_by_zero_bits():
    # "aGVsbG8=" is "hello", but the "=" is dropped and the length is padded
    # with "A" (zero bits), so one extra zero byte follows. It is harmless for
    # arithmetic coded data because the decoder reads zeros past the end.
    assert robust_b64decode(b"aGVsbG8=") == b"hello\x00"


def test_valid_unpadded_multiple_of_four_is_exact():
    assert robust_b64decode(b"aGVsbG8h") == b"hello!"


def test_whitespace_and_garbage_are_ignored():
    assert robust_b64decode(b" a G\tV\ns\r\nb G 8 h ") == b"hello!"
    assert robust_b64decode(b"\xffaGVs\x00bG8h\xfe") == b"hello!"


def test_empty_and_all_invalid_input_give_empty_output():
    assert robust_b64decode(b"") == b""
    assert robust_b64decode(b"=-_!? \n") == b""


def test_urlsafe_characters_are_dropped_not_translated():
    assert robust_b64decode(b"-_-_") == b""


@pytest.mark.parametrize("size", range(0, 40))
def test_round_trip_with_trailing_zero_allowance(size):
    data = bytes((i * 37 + 5) % 256 for i in range(size))
    decoded = robust_b64decode(base64.b64encode(data))
    assert decoded.startswith(data)
    assert set(decoded[len(data):]) <= {0}
    assert len(decoded) - len(data) <= 2


def test_accepts_bytes_from_a_bytearray():
    assert robust_b64decode(bytearray(b"aGVsbG8h")) == b"hello!"


def test_module_has_no_stray_alphabet_difference():
    assert llama_zip.BASE64.decode() == (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    )
