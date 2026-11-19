// Output sinks over the txiki.js standard streams.
//
// tjs.stdout and tjs.stderr are WritableStreams, so a write returns a promise.
// A Sink queues writes in order, takes strings or bytes, and flush() waits for
// everything written so far. Tests pass their own objects with the same two
// methods instead.

export class StreamSink {
    #writer;
    #queue = Promise.resolve();
    #encoder = new TextEncoder();

    constructor(stream) {
        this.#writer = stream.getWriter();
    }

    write(data) {
        const bytes = typeof data === 'string' ? this.#encoder.encode(data) : data;
        this.#queue = this.#queue.then(() => this.#writer.write(bytes));
        return this.#queue;
    }

    flush() {
        return this.#queue;
    }
}

let shared = null;

// One pair of sinks per process, because a stream only gives out one writer.
export const defaultIo = () => shared ??= { out: new StreamSink(tjs.stdout), err: new StreamSink(tjs.stderr) };
