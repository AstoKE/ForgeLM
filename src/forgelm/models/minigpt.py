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

import torch

from forgelm.models.block import LayerNorm, TransformerBlock
from forgelm.models.neural_bigram import cross_entropy


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
        """ids: (T,) token ids -> (logits (T, vocab_size), attention weights per block)."""
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError(f"expected ids of shape (T,) with T >= 1, got {tuple(ids.shape)}")
        if len(ids) > self.block_size:
            raise ValueError(
                f"sequence of {len(ids)} tokens is longer than block_size {self.block_size}: "
                "there is no seat number that high in the position table"
            )
        # Row lookup, exactly like NeuralBigram's W[prev]: cheaper than one_hot @ table.
        x = self.token_embedding[ids] + self.position_embedding[: len(ids)]
        weights = []
        for block in self.blocks:
            x, block_weights = block.forward(x)
            weights.append(block_weights)
        return self.ln_final.forward(x) @ self.head, weights

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over the sequence, reusing 2b's hand-written version."""
        if ids.shape != targets.shape:
            raise ValueError(f"ids {tuple(ids.shape)} and targets {tuple(targets.shape)} differ")
        logits, _ = self.forward(ids)
        return cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> list[int]:
        """Extend `prompt_ids` one sampled token at a time. temperature=0 means greedy.

        Unlike the bigram, this model reads the whole past -- but only the last
        `block_size` tokens of it, because that is all the position table has seats for.
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
                probs = torch.softmax(last / temperature, dim=-1)
                ids.append(int(torch.multinomial(probs, 1, generator=generator)))
        return ids
