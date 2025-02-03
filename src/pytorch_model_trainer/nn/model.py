from dataclasses import dataclass
from typing import Literal

import torch

from pytorch_model_trainer.nn.attention import AttentionBlock


@dataclass
class ModelSettings:
    n_embeddings: int
    max_context_size: int
    num_attention_heads: int
    feed_forward_mult: int
    p_dropout: float
    ff_nonlinearity: Literal["relu", "gelu"]
    num_attention_blocks: int


class Model(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        settings: ModelSettings,
    ) -> None:
        super().__init__()
        self._token_embedding_table = torch.nn.Embedding(
            vocab_size, settings.n_embeddings
        )
        self._position_embedding_table = torch.nn.Embedding(
            settings.max_context_size, settings.n_embeddings
        )
        self._attention_blocks = torch.nn.Sequential(
            *[
                AttentionBlock(
                    settings.max_context_size,
                    settings.n_embeddings,
                    settings.p_dropout,
                    settings.num_attention_heads,
                    settings.p_dropout,
                    settings.feed_forward_mult,
                    settings.ff_nonlinearity,
                    settings.p_dropout,
                )
                for _ in range(settings.num_attention_blocks)
            ],
        )
        self._final_layer_norm = torch.nn.LayerNorm(settings.n_embeddings)
        self._final_linear = torch.nn.Linear(settings.n_embeddings, vocab_size)
        self._max_context_size = settings.max_context_size

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        inputs = inputs[:, -self._max_context_size :]
        batch_size, context_size = inputs.shape
        # inputs: (batch_size, context_size)
        # targets: (batch_size, context_size)

        token_embeddings = self._token_embedding_table(inputs)
        position_embeddings = self._position_embedding_table(
            torch.arange(context_size, device=inputs.device)
        )
        x = token_embeddings + position_embeddings
        # x is (batch_size, context_size, n_embeddings)
        x = self._attention_blocks(x)
        logits: torch.Tensor = self._final_linear(self._final_layer_norm(x))
        # # logits is (batch_size, context_size, vocab_size)
        return logits
