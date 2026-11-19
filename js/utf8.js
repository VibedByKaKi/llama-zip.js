// UTF-8 helpers, a port of Utf8Chunks, bytes_to_utf8 and utf8_to_bytes in llama_zip.py.
//
// Arbitrary bytes are made into valid UTF-8 text before tokenizing. Bytes that
// are not part of a valid UTF-8 sequence become the private use characters
// U+E000 + byte, and real characters in U+E000..U+E0FF are escaped the same
// way, byte by byte, so the mapping can be undone without ambiguity.

export const PUA_START = 0xE000;

const TAG_CONT_U8 = 128;

// Based on Rust's std::str::Utf8Chunks. Iterating yields { valid, invalid }
// objects (Utf8Chunk) holding Uint8Array slices, the longest valid UTF-8 prefix
// of what is left followed by the (possibly empty) run of invalid bytes.
export class Utf8Chunk {
    constructor(valid, invalid) {
        this.valid = valid;
        this.invalid = invalid;
    }
}

export class Utf8Chunks {
    constructor(source) {
        this.source = source;
    }

    [Symbol.iterator]() {
        return this;
    }

    next() {
        if (this.source.length === 0) return { done: true, value: undefined };

        const source = this.source;
        // Reads past the end give 0, which is never a continuation byte.
        const safeGet = index => index < source.length ? source[index] : 0;

        let i = 0;
        let validUpTo = 0;
        while (i < source.length) {
            const byte = source[i];
            i++;

            if (byte < 0x80) {
                // ASCII
            }
            else if (byte >= 0xC2 && byte <= 0xDF) {
                // 2-byte sequence
                if ((safeGet(i) & 192) !== TAG_CONT_U8) break;
                i++;
            }
            else if (byte >= 0xE0 && byte <= 0xEF) {
                // 3-byte sequence
                const next = safeGet(i);
                if (byte === 0xE0 && next >= 0xA0 && next <= 0xBF) {
                    // ok
                }
                else if (byte >= 0xE1 && byte <= 0xEC && next >= 0x80 && next <= 0xBF) {
                    // ok
                }
                else if (byte === 0xED && next >= 0x80 && next <= 0x9F) {
                    // ok
                }
                else if (byte >= 0xEE && byte <= 0xEF && next >= 0x80 && next <= 0xBF) {
                    // ok
                }
                else break;
                i++;
                if ((safeGet(i) & 192) !== TAG_CONT_U8) break;
                i++;
            }
            else if (byte >= 0xF0 && byte <= 0xF4) {
                // 4-byte sequence
                const next = safeGet(i);
                if (byte === 0xF0 && next >= 0x90 && next <= 0xBF) {
                    // ok
                }
                else if (byte >= 0xF1 && byte <= 0xF3 && next >= 0x80 && next <= 0xBF) {
                    // ok
                }
                else if (byte === 0xF4 && next >= 0x80 && next <= 0x8F) {
                    // ok
                }
                else break;
                i++;
                if ((safeGet(i) & 192) !== TAG_CONT_U8) break;
                i++;
                if ((safeGet(i) & 192) !== TAG_CONT_U8) break;
                i++;
            }
            else break;

            validUpTo = i;
        }

        const inspected = source.subarray(0, i);
        this.source = source.subarray(i);
        return { done: false, value: new Utf8Chunk(inspected.slice(0, validUpTo), inspected.slice(validUpTo)) };
    }
}

// UTF-8 bytes of the private use character U+E000 + byte.
const pushPua = (output, byte) => {
    output.push(0xEE, 0x80 | (byte >> 6), 0x80 | (byte & 0x3F));
};

export const bytesToUtf8 = data => {
    const output = [];
    for (const chunk of new Utf8Chunks(data)) {
        const valid = chunk.valid;
        let i = 0;
        while (i < valid.length) {
            const byte = valid[i];
            // U+E000..U+E0FF is EE 80 80 .. EE 83 BF. The chunk is valid UTF-8, so two more bytes follow the lead byte.
            if (byte === 0xEE && valid[i + 1] >= 0x80 && valid[i + 1] <= 0x83) {
                pushPua(output, 0xEE);
                pushPua(output, valid[i + 1]);
                pushPua(output, valid[i + 2]);
                i += 3;
            }
            else {
                output.push(byte);
                i++;
            }
        }
        for (const byte of chunk.invalid) pushPua(output, byte);
    }
    return Uint8Array.from(output);
};

export const utf8ToBytes = text => {
    const output = [];
    const encoder = new TextEncoder();
    for (const char of text) {
        const codePoint = char.codePointAt(0);
        if (codePoint >= PUA_START && codePoint <= PUA_START + 0xFF) output.push(codePoint - PUA_START);
        else if (codePoint < 0x80) output.push(codePoint);
        else output.push(...encoder.encode(char));
    }
    return Uint8Array.from(output);
};
