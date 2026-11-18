"""Deterministic stand-in for llama_cpp.Llama.

The model is a byte-level tokenizer with a few merges and a logits function that
depends only on the context. Both are plain integer arithmetic so the same model
can be written in another language (see tests/vectors/fake_model.json).
"""

import numpy as np

BOS = 0
EOS = 1
BYTE_BASE = 2
MERGES = [b" the", b"the", b"ing", b"he", b"th", b"  ", b" "]
DEFAULT_N_CTX = 64


def build_vocab():
    pieces = [b"", b""] + [bytes([i]) for i in range(256)] + list(MERGES)
    return pieces


VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)


def fake_logits(context):
    last = context[-1] if context else 0
    before = context[-2] if len(context) > 1 else 0
    h = last * 131 + before * 31 + len(context) * 7
    return np.array(
        [((h * (i + 3) + i * i) % 31) / 2 for i in range(VOCAB_SIZE)],
        dtype=np.float32,
    )


class FakeLlama:
    instances = []
    space_prefix = False
    on_logits = None

    def __init__(
        self, model_path, use_mlock=False, n_gpu_layers=-1, n_ctx=0, verbose=True
    ):
        self.model_path = model_path
        self.use_mlock = use_mlock
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._n_ctx = n_ctx or DEFAULT_N_CTX
        self.reset_calls = 0
        self.generate_calls = []
        FakeLlama.instances.append(self)

    def n_ctx(self):
        return self._n_ctx

    def token_bos(self):
        return BOS

    def token_eos(self):
        return EOS

    def reset(self):
        self.reset_calls += 1

    def tokenize(self, text, add_bos=True, special=False):
        if self.space_prefix and text:
            text = b" " + text
        tokens = [BOS] if add_bos else []
        by_length = sorted(
            range(len(MERGES)), key=lambda k: len(MERGES[k]), reverse=True
        )
        i = 0
        while i < len(text):
            for k in by_length:
                if text.startswith(MERGES[k], i):
                    tokens.append(BYTE_BASE + 256 + k)
                    i += len(MERGES[k])
                    break
            else:
                tokens.append(BYTE_BASE + text[i])
                i += 1
        return tokens

    def detokenize(self, tokens):
        return b"".join(VOCAB[t] for t in tokens)

    def logits_to_logprobs(self, logits):
        x = np.asarray(logits, dtype=np.float64)
        shifted = x - np.max(x)
        return shifted - np.log(np.sum(np.exp(shifted)))

    def generate(self, tokens, temp=0.0, logits_processor=None, stopping_criteria=None):
        assert temp == 0.0
        context = list(tokens)
        self.generate_calls.append(list(tokens))
        if len(context) > self._n_ctx:
            raise ValueError("Requested tokens exceed context window")
        while True:
            scores = fake_logits(context)
            if FakeLlama.on_logits is not None:
                FakeLlama.on_logits(list(context))
            for processor in logits_processor or []:
                scores = processor(np.array(context), scores)
            token = int(np.argmax(scores))
            if stopping_criteria is not None and stopping_criteria(
                np.array(context), scores
            ):
                return
            yield token
            context.append(token)
