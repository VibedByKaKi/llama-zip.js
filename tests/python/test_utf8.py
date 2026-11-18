import random

import pytest

from llama_zip import PUA_START, Utf8Chunk, Utf8Chunks, bytes_to_utf8, utf8_to_bytes
from vectors import case_ids, load_vectors

VECTORS = load_vectors("utf8.json")


def chunks_of(data):
    return [(c.valid, c.invalid) for c in Utf8Chunks(data)]


@pytest.mark.parametrize("case", VECTORS["chunks"], ids=case_ids(VECTORS["chunks"]))
def test_utf8_chunks_vectors(case):
    expected = [
        (bytes.fromhex(c["valid_hex"]), bytes.fromhex(c["invalid_hex"]))
        for c in case["chunks"]
    ]
    assert chunks_of(bytes.fromhex(case["input_hex"])) == expected


@pytest.mark.parametrize("case", VECTORS["chunks"], ids=case_ids(VECTORS["chunks"]))
def test_utf8_chunks_cover_the_input_exactly(case):
    data = bytes.fromhex(case["input_hex"])
    assert b"".join(v + i for v, i in chunks_of(data)) == data


@pytest.mark.parametrize("case", VECTORS["chunks"], ids=case_ids(VECTORS["chunks"]))
def test_utf8_chunks_valid_part_decodes(case):
    for valid, _ in chunks_of(bytes.fromhex(case["input_hex"])):
        valid.decode("utf-8")


def test_utf8_chunks_empty_input_yields_nothing():
    assert list(Utf8Chunks(b"")) == []


def test_utf8_chunks_is_its_own_iterator():
    chunks = Utf8Chunks(b"a\xff")
    assert iter(chunks) is chunks
    assert isinstance(next(chunks), Utf8Chunk)


def test_utf8_chunks_matches_python_for_valid_text():
    text = "aé€😀\U0010ffff퟿ end"
    assert chunks_of(text.encode()) == [(text.encode(), b"")]


def test_utf8_chunks_valid_prefix_agrees_with_python_decoder():
    rng = random.Random(5)
    for _ in range(500):
        data = bytes(rng.choice([0x41, 0x80, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98, 0x80, 0xFF, 0xED, 0xA0])
                     for _ in range(rng.randint(0, 12)))
        rebuilt = b""
        for valid, invalid in chunks_of(data):
            assert valid.decode("utf-8") is not None
            if invalid:
                # The invalid run must really be undecodable as a lone chunk.
                with pytest.raises(UnicodeDecodeError):
                    invalid.decode("utf-8")
            rebuilt += valid + invalid
        assert rebuilt == data
        bytes_to_utf8(data).decode("utf-8")


@pytest.mark.parametrize(
    "case", VECTORS["bytes_to_utf8"], ids=case_ids(VECTORS["bytes_to_utf8"])
)
def test_bytes_to_utf8_vectors(case):
    result = bytes_to_utf8(bytes.fromhex(case["input_hex"]))
    assert result.hex() == case["output_hex"]
    result.decode("utf-8")


def test_bytes_to_utf8_passes_ordinary_text_through():
    text = "plain text é € 😀"
    assert bytes_to_utf8(text.encode()) == text.encode()


def test_bytes_to_utf8_maps_invalid_bytes_into_the_pua():
    assert bytes_to_utf8(b"\xff").decode() == chr(PUA_START + 0xFF)
    assert bytes_to_utf8(b"\x80").decode() == chr(PUA_START + 0x80)
    assert bytes_to_utf8(b"a\xc3").decode() == "a" + chr(PUA_START + 0xC3)


def test_bytes_to_utf8_escapes_real_pua_characters_byte_by_byte():
    # U+E000 is EE 80 80 in UTF-8, so it must come back as three PUA characters.
    assert bytes_to_utf8("".encode()).decode() == "".join(
        chr(PUA_START + b) for b in (0xEE, 0x80, 0x80)
    )
    # U+E100 is outside the mapped range and stays as is.
    assert bytes_to_utf8("".encode()) == "".encode()


def test_pua_boundaries():
    assert bytes_to_utf8("".encode()) != "".encode()
    assert bytes_to_utf8("".encode()) == "".encode()
    assert bytes_to_utf8("\udfff".encode("utf-8", "surrogatepass")) == "".join(
        chr(PUA_START + b) for b in (0xED, 0xBF, 0xBF)
    ).encode()


@pytest.mark.parametrize(
    "case", VECTORS["utf8_to_bytes"], ids=case_ids(VECTORS["utf8_to_bytes"])
)
def test_utf8_to_bytes_vectors(case):
    text = "".join(chr(c) for c in case["codepoints"])
    assert utf8_to_bytes(text).hex() == case["output_hex"]


def test_utf8_to_bytes_maps_pua_to_single_bytes():
    assert utf8_to_bytes(chr(PUA_START)) == b"\x00"
    assert utf8_to_bytes(chr(PUA_START + 0xFF)) == b"\xff"
    assert utf8_to_bytes("") == "".encode()


def test_utf8_to_bytes_empty():
    assert utf8_to_bytes("") == b""


@pytest.mark.parametrize(
    "case", VECTORS["roundtrip"], ids=case_ids(VECTORS["roundtrip"])
)
def test_round_trip_vectors(case):
    data = bytes.fromhex(case["input_hex"])
    assert utf8_to_bytes(bytes_to_utf8(data).decode("utf-8")) == data


def test_round_trip_every_single_byte():
    for value in range(256):
        data = bytes([value])
        assert utf8_to_bytes(bytes_to_utf8(data).decode("utf-8")) == data


def test_round_trip_every_pair_of_bytes_sampled():
    rng = random.Random(11)
    for _ in range(3000):
        data = bytes([rng.randrange(256), rng.randrange(256)])
        assert utf8_to_bytes(bytes_to_utf8(data).decode("utf-8")) == data


def test_round_trip_random_bytes():
    rng = random.Random(42)
    for _ in range(300):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
        assert utf8_to_bytes(bytes_to_utf8(data).decode("utf-8")) == data


def test_round_trip_random_text_with_pua():
    rng = random.Random(43)
    alphabet = ["a", "é", "", "", "", "😀", "€"]
    for _ in range(100):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        data = text.encode()
        assert utf8_to_bytes(bytes_to_utf8(data).decode("utf-8")) == data


def test_bytes_to_utf8_is_injective_on_pua_collisions():
    assert bytes_to_utf8(b"\xff") != bytes_to_utf8("".encode())
