// Minimal test runner for txiki.js, reusable by later suites.
//
//   import { test, assert, run } from './run.js';
//   test('adds', () => assert.equal(1 + 1, 2));
//   await run();
//
// Tests run in order, one at a time, and may be async. run() prints one line
// per test (skip() registers a skipped one), exits the process with code 1 if any failed, and returns otherwise.
// Use `tjs run js/test/<name>.test.js`.
const tests = [];

export const test = (name, fn) => tests.push({ name, fn });

// Registers a test that is reported as skipped, with the reason.
export const skip = (name, reason) => tests.push({ name, skip: reason });

export const info = message => console.log(`      ${message}`);

const show = value => {
    try {
        return typeof value === 'bigint' ? `${value}n` : JSON.stringify(value) ?? String(value);
    }
    catch {
        return String(value);
    }
};

export const assert = {
    ok(value, message = 'expected a truthy value') {
        if (!value) throw new Error(message);
    },
    equal(actual, expected, message = '') {
        if (actual !== expected) throw new Error(`${message} expected ${show(expected)}, got ${show(actual)}`.trim());
    },
    deepEqual(actual, expected, message = '') {
        const a = show(Array.from(actual));
        const b = show(Array.from(expected));
        if (a !== b) throw new Error(`${message} expected ${b}, got ${a}`.trim());
    },
    notEqual(actual, expected, message = '') {
        if (actual === expected) throw new Error(`${message} expected a value other than ${show(expected)}`.trim());
    },
    async rejects(fn, pattern, message = 'expected a rejection') {
        try {
            await fn();
        }
        catch (error) {
            if (pattern && !pattern.test(String(error?.message ?? error))) throw new Error(`${message}: rejected with "${error?.message}"`);
            return;
        }
        throw new Error(message);
    },
    throws(fn, pattern, message = 'expected a throw') {
        try {
            fn();
        }
        catch (error) {
            if (pattern && !pattern.test(String(error?.message ?? error))) throw new Error(`${message}: threw "${error?.message}"`);
            return;
        }
        throw new Error(message);
    },
};

export const run = async () => {
    let failed = 0;
    let skipped = 0;
    for (const { name, fn, skip: reason } of tests) {
        if (reason !== undefined) {
            skipped++;
            console.log(`skip   ${name} (${reason})`);
            continue;
        }
        const start = Date.now();
        try {
            await fn();
            console.log(`ok     ${name} (${Date.now() - start} ms)`);
        }
        catch (error) {
            failed++;
            console.log(`FAIL   ${name}\n      ${error?.stack ?? error}`);
        }
    }
    console.log(`\n${tests.length - failed - skipped}/${tests.length} passed${skipped ? `, ${skipped} skipped` : ''}`);
    if (failed) tjs.exit(1);
};
