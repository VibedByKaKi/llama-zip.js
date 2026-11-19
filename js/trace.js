// Step objects for observing the arithmetic coder, used by LlamaZip's onStep option and trace().
//
// A step describes one coded token. It holds only strings, numbers, booleans, arrays and plain
// objects, so JSON.stringify works on it as it is (no BigInt, no typed arrays).
//
// Nothing here feeds back into the coder. The functions read the cdf and the coder state after
// the symbol is coded, so the compressed bytes are the same with and without observers.
import { NUM_STATE_BITS } from './arithmetic.js';

export const DEFAULT_TOP_K = 16;
export const DEFAULT_OUTPUT_BITS_CAP = 256;

const HEX_DIGITS = NUM_STATE_BITS / 4;

export const toHex64 = value => value.toString(16).padStart(HEX_DIGITS, '0');

// The bits [from, to) of an Encoder's output as a string of 0 and 1.
export const readBits = (encoder, from, to) => {
    let text = '';
    for (let i = from; i < to; i++) text += (encoder.encodedData[i >> 3] >> (7 - (i & 7))) & 1;
    return text;
};

// The k tokens with the most probability, most probable first (lower id first on a tie), found
// in one pass with an insertion into a list of k entries, not a sort of the whole vocabulary.
// Most tokens fail the first comparison once the list is full. Returns the token ids.
export const selectTopTokens = (cdf, k) => {
    const n = cdf.length;
    const size = Math.min(k, n);
    const ids = new Int32Array(size);
    const frequencies = new Float64Array(size);
    let count = 0;
    let previous = 0;
    // Smallest frequency in the list once it is full. Until then nothing is skipped.
    let floor = -1;
    for (let i = 0; i < n; i++) {
        const cumulative = cdf[i];
        const frequency = cumulative - previous;
        previous = cumulative;
        if (frequency <= floor) continue;
        if (size === 0) break;
        let j = count < size ? count++ : size - 1;
        while (j > 0 && frequencies[j - 1] < frequency) {
            frequencies[j] = frequencies[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        frequencies[j] = frequency;
        ids[j] = i;
        if (count === size) floor = frequencies[size - 1];
    }
    return Array.from(ids.subarray(0, count));
};

const lenientDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
const utf8Encoder = new TextEncoder();

const sameBytes = (a, b) => a.length === b.length && a.every((byte, i) => byte === b[i]);

export class StepBuilder {
    // model needs detokenize() and tokenEos(). encoder is the Encoder whose state the steps
    // report (the decoder side passes a shadow encoder that codes the same symbols).
    constructor(model, encoder, { topK = DEFAULT_TOP_K, outputBitsCap = DEFAULT_OUTPUT_BITS_CAP } = {}) {
        if (!Number.isInteger(topK) || topK < 0) throw new RangeError(`topK must be a non-negative integer, got ${topK}`);
        if (!Number.isInteger(outputBitsCap) || outputBitsCap < 0) throw new RangeError(`outputBitsCap must be a non-negative integer, got ${outputBitsCap}`);
        this.model = model;
        this.encoder = encoder;
        this.topK = topK;
        this.outputBitsCap = outputBitsCap;
        this.eos = model.tokenEos();
        this.pieces = new Map();
        this.cumulativeBits = 0;
        this.emittedBefore = 0;
    }

    // The text of a token decoded leniently, and its bytes when that decoding loses something.
    piece(tokenId) {
        let entry = this.pieces.get(tokenId);
        if (!entry) {
            const bytes = tokenId === this.eos ? new Uint8Array(0) : this.model.detokenize([tokenId]);
            const piece = lenientDecoder.decode(bytes);
            const lossy = !sameBytes(utf8Encoder.encode(piece), bytes);
            entry = { piece, pieceBytes: lossy ? Array.from(bytes) : null };
            this.pieces.set(tokenId, entry);
        }
        return entry;
    }

    // Call after the encoder has coded `tokenId` with `cdf`.
    // position: { index, windowIndex, windowReset, contextLength }.
    build(cdf, tokenId, position) {
        const total = cdf[cdf.length - 1];
        const low = tokenId > 0 ? cdf[tokenId - 1] : 0;
        const frequency = cdf[tokenId] - low;
        const probability = frequency / total;
        const bits = -Math.log2(probability);
        this.cumulativeBits += bits;

        const listed = selectTopTokens(cdf, this.topK);
        if (!listed.includes(tokenId)) listed.push(tokenId);
        const frequencyOf = id => cdf[id] - (id > 0 ? cdf[id - 1] : 0);
        listed.sort((a, b) => frequencyOf(b) - frequencyOf(a) || a - b);
        let listedFrequency = 0;
        const candidates = listed.map(id => {
            const candidateFrequency = frequencyOf(id);
            listedFrequency += candidateFrequency;
            return {
                tokenId: id,
                piece: this.piece(id).piece,
                p: candidateFrequency / total,
                lo: (id > 0 ? cdf[id - 1] : 0) / total,
                hi: cdf[id] / total,
                chosen: id === tokenId,
            };
        });

        const encoder = this.encoder;
        const emittedBits = encoder.bitCount;
        const { piece, pieceBytes } = this.piece(tokenId);
        const step = {
            index: position.index,
            tokenId,
            piece,
            pieceBytes,
            isEos: tokenId === this.eos,
            probability,
            bits,
            cumulativeBits: this.cumulativeBits,
            lo: low / total,
            hi: cdf[tokenId] / total,
            candidates,
            otherProbability: (total - listedFrequency) / total,
            emittedBits,
            newBits: readBits(encoder, this.emittedBefore, emittedBits),
            outputBits: readBits(encoder, Math.max(0, emittedBits - this.outputBitsCap), emittedBits),
            pendingUnderflow: encoder.numUnderflow,
            coderLow: toHex64(encoder.low),
            coderHigh: toHex64(encoder.high),
            windowIndex: position.windowIndex,
            windowReset: position.windowReset,
            contextLength: position.contextLength,
        };
        this.emittedBefore = emittedBits;
        return step;
    }
}
