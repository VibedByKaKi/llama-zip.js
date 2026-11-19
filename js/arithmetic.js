// Arithmetic coder, a port of the Encoder and Decoder in llama_zip.py.
//
// The coder keeps a 64-bit interval [low, high]. A JavaScript number is a
// double and holds integers exactly only up to 2^53, so the interval and the
// code word are BigInt. Cumulative frequencies stay plain numbers: the
// total is about 2^32 (see FREQ_SCALE_FACTOR in llamazip.js), far below 2^53,
// and are converted with BigInt() only where they meet the 64-bit state.
// The bit stream is the same as the Python tool writes.
//
// A cdf is an array-like of non-decreasing integers (Array, Float64Array...)
// where cdf[i] is the cumulative frequency up to and including symbol i.

export const NUM_STATE_BITS = 64;

export class ArithmeticCoderBase {
    constructor() {
        const fullRange = 1n << BigInt(NUM_STATE_BITS);
        this.halfRange = fullRange >> 1n;
        this.quarterRange = this.halfRange >> 1n;
        this.stateMask = fullRange - 1n;
        this.low = 0n;
        this.high = this.stateMask;
    }

    update(cdf, symbol) {
        const total = BigInt(cdf[cdf.length - 1]);
        const range = this.high - this.low + 1n;
        const symHigh = BigInt(cdf[symbol]);
        this.high = this.low + symHigh * range / total - 1n;
        const symLow = symbol > 0 ? BigInt(cdf[symbol - 1]) : 0n;
        this.low = this.low + symLow * range / total;
        while (((this.low ^ this.high) & this.halfRange) === 0n) {
            this.shift();
            this.low = (this.low << 1n) & this.stateMask;
            this.high = ((this.high << 1n) & this.stateMask) | 1n;
        }
        while ((this.low & ~this.high & this.quarterRange) !== 0n) {
            this.underflow();
            this.low = (this.low << 1n) ^ this.halfRange;
            this.high = ((this.high ^ this.halfRange) << 1n) | this.halfRange | 1n;
        }
    }

    shift() {
        throw new Error('not implemented');
    }

    underflow() {
        throw new Error('not implemented');
    }
}

export class Encoder extends ArithmeticCoderBase {
    constructor() {
        super();
        this.encodedData = new Uint8Array(64);
        this.length = 0;
        this.bitIndex = 8;
        this.numUnderflow = 0;
    }

    // The encoded bytes so far. The last byte may be partly filled.
    getEncoded() {
        return this.encodedData.slice(0, this.length);
    }

    encodeSymbol(cdf, symbol) {
        this.update(cdf, symbol);
    }

    finish() {
        this.appendBit(1);
    }

    shift() {
        const bit = Number(this.low >> BigInt(NUM_STATE_BITS - 1));
        this.appendBit(bit);
        for (let i = 0; i < this.numUnderflow; i++) this.appendBit(bit ^ 1);
        this.numUnderflow = 0;
    }

    underflow() {
        this.numUnderflow++;
    }

    appendBit(bit) {
        if (this.bitIndex === 8) {
            if (this.length === this.encodedData.length) {
                const grown = new Uint8Array(this.encodedData.length * 2);
                grown.set(this.encodedData);
                this.encodedData = grown;
            }
            this.encodedData[this.length++] = 0;
            this.bitIndex = 0;
        }
        this.encodedData[this.length - 1] |= bit << (7 - this.bitIndex);
        this.bitIndex++;
    }
}

export class Decoder extends ArithmeticCoderBase {
    constructor(data) {
        super();
        this.input = data;
        this.byteIndex = 0;
        this.bitIndex = 0;
        // Bits requested so far, including the zero bits supplied past the end of the data.
        this.bitsRead = 0;
        this.code = 0n;
        for (let i = 0; i < NUM_STATE_BITS; i++) this.code = (this.code << 1n) | BigInt(this.readCodeBit());
    }

    decodeSymbol(cdf) {
        const total = BigInt(cdf[cdf.length - 1]);
        const range = this.high - this.low + 1n;
        const offset = this.code - this.low;
        // value < total, and total is below 2^53, so the conversion is exact.
        const value = Number(((offset + 1n) * total - 1n) / range);
        // First index whose cumulative frequency is greater than value (numpy searchsorted, side="right").
        let lo = 0;
        let hi = cdf.length;
        while (lo < hi) {
            const mid = (lo + hi) >>> 1;
            if (cdf[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        this.update(cdf, lo);
        return lo;
    }

    shift() {
        this.code = ((this.code << 1n) & this.stateMask) | BigInt(this.readCodeBit());
    }

    underflow() {
        this.code = (this.code & this.halfRange)
            | ((this.code << 1n) & (this.stateMask >> 1n))
            | BigInt(this.readCodeBit());
    }

    readCodeBit() {
        this.bitsRead++;
        if (this.byteIndex >= this.input.length) return 0;
        const bit = (this.input[this.byteIndex] >> (7 - this.bitIndex)) & 1;
        this.bitIndex = (this.bitIndex + 1) % 8;
        if (this.bitIndex === 0) this.byteIndex++;
        return bit;
    }
}
