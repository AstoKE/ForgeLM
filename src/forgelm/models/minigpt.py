"""MiniGPT: the whole model, from token ids to logits.

    ids (T,) -> token_embedding + position_embedding -> block x n -> ln_final -> @ head
             -> logits (T, V)

Two embedding tables feed the stack:

* **token embedding**, one row per vocabulary entry: "what is this token?"
* **position embedding**, one row per seat: "where am I in the sequence?"

The second one is necessary because attention scores `q . k` are built from content
only, so "the dog bit the man" and "the man bit the dog" would look the same. The
causal mask does carry a little order ("5 tokens came before me") but not which token
sat where. The position table fixes that, and it is learned like everything else.

Its height is also the model's hard limit: `block_size` rows means no seat number
`block_size + 1` exists, which is exactly what a "context window" is.

At the end a final LayerNorm and one `(C, V)` matrix turn each position into a score
per vocabulary entry -- the logits, the same thing the bigram produced, except this
model looked at the whole past instead of one token.

Requires the optional `ml` extra (torch).
"""

from pathlib import Path

import torch

from forgelm.models.block import LayerNorm, TransformerBlock
from forgelm.models.neural_bigram import cross_entropy
from forgelm.tokenizer import CharTokenizer

CHECKPOINT_FORMAT_VERSION = 1


class MiniGPT:
    """A small GPT: embeddings, a stack of TransformerBlocks, and a head to the vocab."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        block_size: int = 128,
        hidden_dim: int | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if vocab_size < 1 or block_size < 1 or num_blocks < 1:
            raise ValueError("vocab_size, block_size and num_blocks must be >= 1")
        gen = torch.Generator().manual_seed(seed)

        def table(rows: int, cols: int) -> torch.Tensor:
            scaled = torch.randn((rows, cols), generator=gen) * cols**-0.5
            return scaled.to(device).requires_grad_(True)

        self.token_embedding = table(vocab_size, embed_dim)  # "what is this token?"
        self.position_embedding = table(block_size, embed_dim)  # "which seat am I in?"
        # Each block gets its own seed range; a block owns num_heads + 2 seeds.
        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                hidden_dim,
                device=device,
                seed=seed + 1 + i * (num_heads + 2),
            )
            for i in range(num_blocks)
        ]
        self.ln_final = LayerNorm(embed_dim, device=device)
        self.head = table(embed_dim, vocab_size)  # C -> V: one score per vocabulary entry

    @property
    def vocab_size(self) -> int:
        return self.token_embedding.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.token_embedding.shape[1]

    @property
    def block_size(self) -> int:
        """The longest sequence the model can see: the height of the position table."""
        return self.position_embedding.shape[0]

    def config(self) -> dict:
        """The hyperparameters that rebuild an empty model of the same shape."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.blocks[0].attention.num_heads,
            "num_blocks": len(self.blocks),
            "block_size": self.block_size,
            "hidden_dim": self.blocks[0].feedforward.W1.shape[1],
        }

    def parameters(self) -> list[torch.Tensor]:
        return [
            self.token_embedding,
            self.position_embedding,
            *(p for block in self.blocks for p in block.parameters()),
            *self.ln_final.parameters(),
            self.head,
        ]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ids: (T,) or (B, T) -> (logits (..., T, vocab_size), attention weights per block).

        A batch costs almost nothing extra on a GPU: the same kernels run once over B
        windows instead of B times over one. Everything below indexes the last axis, so
        the batch axis just rides along.
        """
        if ids.ndim not in (1, 2) or ids.shape[-1] == 0:
            raise ValueError(
                f"expected ids of shape (T,) or (B, T) with T >= 1, got {tuple(ids.shape)}"
            )
        length = ids.shape[-1]
        if length > self.block_size:
            raise ValueError(
                f"sequence of {length} tokens is longer than block_size {self.block_size}: "
                "there is no seat number that high in the position table"
            )
        # Row lookup, exactly like NeuralBigram's W[prev]: cheaper than one_hot @ table.
        # (B, T) ids give (B, T, C); the (T, C) position table broadcasts over the batch.
        x = self.token_embedding[ids] + self.position_embedding[:length]
        weights = []
        for block in self.blocks:
            x, block_weights = block.forward(x)
            weights.append(block_weights)
        return self.ln_final.forward(x) @ self.head, weights

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over every position, reusing 2b's hand-written version.

        `cross_entropy` wants (N, V) and (N,), so a batch is flattened: B windows of T
        positions are simply B * T predictions, all weighted the same.
        """
        if ids.shape != targets.shape:
            raise ValueError(f"ids {tuple(ids.shape)} and targets {tuple(targets.shape)} differ")
        logits, _ = self.forward(ids)
        return cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        stop_id: int | None = None,
    ) -> list[int]:
        """Extend `prompt_ids` one sampled token at a time. temperature=0 means greedy.

        Unlike the bigram, this model reads the whole past -- but only the last
        `block_size` tokens of it, because that is all the position table has seats for.

        With `stop_id`, generation ends right after that token is produced (it is kept in
        the result). An answer that ends at a newline then costs a few steps, not
        `max_new_tokens`.
        """
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        device = self.token_embedding.device
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = ids[-self.block_size :]  # crop: the oldest tokens fall out of view
            logits, _ = self.forward(torch.tensor(context, device=device))
            last = logits[-1]  # only the prediction for the final position matters
            if temperature == 0:
                ids.append(int(last.argmax()))
            else:
                # The seeded generator lives on the CPU, and a vocabulary-sized vector is
                # nothing to move, so sampling is the same whichever device the model is on.
                probs = torch.softmax(last / temperature, dim=-1).cpu()
                ids.append(int(torch.multinomial(probs, 1, generator=generator)))
            if stop_id is not None and ids[-1] == stop_id:
                break
        return ids


def save_minigpt(path: str | Path, model: MiniGPT, tokenizer: CharTokenizer) -> None:
    """Save weights, hyperparameters *and* tokenizer in one file (ADR 0005).

    The weights are a plain list of tensors in `model.parameters()` order, loaded back
    with `torch.load(weights_only=True)`, which refuses anything that is not a tensor
    or a basic Python value, so a shared file cannot run code.
    """
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer disagree on vocab size")
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "tokenizer": tokenizer.to_dict(),
        "model": {
            "type": "minigpt",
            "config": model.config(),
            "weights": [p.detach().cpu() for p in model.parameters()],
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_minigpt(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MiniGPT, CharTokenizer]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except OSError:
        raise  # a missing or unreadable file is a different problem from a corrupt one
    except Exception as exc:  # noqa: BLE001
        # torch.load raises a different error type for each way a file can be garbage
        # (UnpicklingError, RuntimeError, EOFError, IndexError...): one clear message instead.
        raise ValueError(f"not a MiniGPT checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("not a MiniGPT checkpoint: expected a dict")
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version: {version!r}")
    saved = checkpoint["model"]
    if saved.get("type") != "minigpt":
        raise ValueError(f"not a minigpt model: type={saved.get('type')!r}")

    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = MiniGPT(**saved["config"], device=device)  # random weights, right shapes
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint model and tokenizer disagree on vocab size")

    weights = saved["weights"]
    params = model.parameters()
    if len(weights) != len(params):
        raise ValueError(f"checkpoint has {len(weights)} tensors, the model needs {len(params)}")
    with torch.no_grad():  # copying numbers in is not something autograd should record
        for i, (param, weight) in enumerate(zip(params, weights, strict=True)):
            if param.shape != weight.shape:
                raise ValueError(
                    f"tensor {i}: checkpoint shape {tuple(weight.shape)}"
                    f" != model shape {tuple(param.shape)}"
                )
            param.copy_(weight)
    return model, tokenizer
