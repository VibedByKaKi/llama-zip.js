// Loads the shared test vectors in tests/vectors/*.json, the same files the Python tests use.
export const loadVectors = async name => {
    const bytes = await tjs.readFile(new URL(`../../tests/vectors/${name}`, import.meta.url).pathname);
    return JSON.parse(new TextDecoder().decode(bytes));
};

export const fromHex = hex => Uint8Array.from(hex.match(/../g) ?? [], pair => parseInt(pair, 16));

export const toHex = bytes => Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');

export const bytesOf = text => new TextEncoder().encode(text);

export const concatBytes = (...parts) => {
    const output = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0;
    for (const part of parts) {
        output.set(part, offset);
        offset += part.length;
    }
    return output;
};

// Small seeded generator (mulberry32). Python's random.Random is not reproducible here, so the
// randomized tests use this one and draw different, but equally fixed, values.
export const makeRng = seed => {
    let state = seed >>> 0;
    const next = () => {
        state = (state + 0x6D2B79F5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
        // Integer in [0, n).
        below: n => Math.floor(next() * n),
        // Integer in [low, high], like random.randint.
        between: (low, high) => low + Math.floor(next() * (high - low + 1)),
        choice: items => items[Math.floor(next() * items.length)],
    };
};
