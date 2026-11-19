// Runs every test module in its own process and prints the totals.
//   tjs run js/test/all.js [name ...]      names like arithmetic, utf8 (default: all)
// Each module is run as `tjs run js/test/<name>.test.js`, so one crash cannot hide the others.
// The real model modules (integration, llama) skip or fail on their own if libllama or the model is missing.
const MODULES = ['arithmetic', 'utf8', 'base64', 'compute_cdf', 'llamazip', 'cli', 'integration', 'llama'];

const selected = tjs.args.slice(3).length ? tjs.args.slice(3) : MODULES;
const root = new URL('.', import.meta.url).pathname;

const readAll = async stream => {
    const reader = stream.getReader();
    let text = '';
    const decoder = new TextDecoder();
    for (;;) {
        const { value, done } = await reader.read();
        if (done) return text + decoder.decode();
        text += decoder.decode(value, { stream: true });
    }
};

let failed = 0;
let passed = 0;
let skipped = 0;
for (const name of selected) {
    const proc = tjs.spawn([tjs.exePath, 'run', `${root}${name}.test.js`], { stdout: 'pipe', stderr: 'pipe' });
    const out = readAll(proc.stdout);
    const err = readAll(proc.stderr);
    const status = await proc.wait();
    const text = await out;
    const lines = text.split('\n');
    const summary = lines.filter(line => /^\d+\/\d+ passed/.test(line)).pop();
    const match = summary && /^(\d+)\/(\d+) passed(?:, (\d+) skipped)?/.exec(summary);
    if (match) {
        passed += Number(match[1]);
        skipped += Number(match[3] ?? 0);
    }
    for (const line of lines) if (line.startsWith('FAIL')) console.log(`${name}: ${line}`);
    if (status.exit_status !== 0 || !match) {
        failed++;
        console.log(`${name}: exit ${status.exit_status}\n${await err}`);
    }
    console.log(`${name.padEnd(12)} ${summary ?? 'no summary'}`);
}
console.log(`\n${passed} passed, ${skipped} skipped, ${failed} module(s) failed`);
if (failed) tjs.exit(1);
