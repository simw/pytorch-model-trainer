from typing import Literal, cast

import torch


class AttentionError(Exception):
    pass


class MultiHeadAttention(torch.nn.Module):
    def __init__(  # noqa: PLR0913 - Allow too many arguments
        self,
        max_context_size: int,
        input_dim: int,
        head_size: int,
        p_dropout: float,
        num_heads: int,
        p_multihead_dropout: float,
    ) -> None:
        super().__init__()
        output_dim = head_size * num_heads
        self._key_query_value = torch.nn.Linear(input_dim, 3 * output_dim, bias=False)
        self._dropout = torch.nn.Dropout(p_dropout)
        self._projection = torch.nn.Linear(output_dim, output_dim)
        self._multi_head_dropout = torch.nn.Dropout(p_multihead_dropout)

        self._num_heads = num_heads
        self._head_size = head_size

        tril = torch.tril(torch.ones(max_context_size, max_context_size))
        self.register_buffer("tril", tril)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        # x is (batch_size, context_size, input_dim)
        # _key_query_value is (input_dim, 3 * head_size * num_heads)

        kqv = self._key_query_value(x).view(
            x.shape[:-1] + (self._num_heads, self._head_size, 3)
        )
        # kqv: (batch_size, context_size, num_heads, head_size, 3)
        keys = kqv[..., 0].transpose(1, 2)
        queries = kqv[..., 1].transpose(1, 2)
        values = kqv[..., 2].transpose(1, 2)
        # keys, queries, values are (batch_size, num_heads, context_size, head_size)

        weights = queries @ keys.transpose(-2, -1) * self._head_size**-0.5
        # weights is (batch_size, num_heads, head_size, head_size)

        context_size = x.shape[-2]
        context_tril = cast(torch.Tensor, self.tril)[:context_size, :context_size]
        weights = weights.masked_fill(context_tril == 0, float("-inf"))
        weights = torch.nn.functional.softmax(weights, dim=-1)
        # weights is still (batch_size, num_heads, head_size, head_size)

        weights = self._dropout(weights)
        result: torch.Tensor = weights @ values
        # result is (batch_size, context_size, head_size)

        output_dim = self._num_heads * self._head_size
        result = (
            result.transpose(1, 2).contiguous().view(input_shape[:-1] + (output_dim,))
        )
        result = self._projection(result)
        result = self._multi_head_dropout(result)
        return result


non_linearities = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
}


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        non_linearity: Literal["relu", "gelu"],
        n_output: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            non_linearities[non_linearity](),
            torch.nn.Linear(n_hidden, n_output),
            torch.nn.Dropout(p_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(  # noqa: PLR0913 - Allow too many arguments
        self,
        max_context_size: int,
        input_dim: int,
        p_attention_dropout: float,
        num_attention_heads: int,
        p_multihead_attention_dropout: float,
        feed_forward_mult: int,
        ff_nonlinearity: Literal["relu", "gelu"],
        p_ffwd_dropout: float,
    ) -> None:
        super().__init__()
        if input_dim % num_attention_heads != 0:
            msg = (
                f"Input dimension {input_dim} should be "
                f"exactly divisible by {num_attention_heads}"
            )
            raise AttentionError(msg)

        head_size = input_dim // num_attention_heads
        self._attention_layer_norm = torch.nn.LayerNorm(input_dim)
        self._attention = MultiHeadAttention(
            max_context_size,
            input_dim,
            head_size,
            p_attention_dropout,
            num_attention_heads,
            p_multihead_attention_dropout,
        )
        self._ffwd_layer_norm = torch.nn.LayerNorm(input_dim)
        attention_output_dim = head_size * num_attention_heads
        self._feed_forward = FeedForward(
            attention_output_dim,
            attention_output_dim * feed_forward_mult,
            ff_nonlinearity,
            input_dim,
            p_ffwd_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attention(self._attention_layer_norm(x))
        x = x + self._feed_forward(self._ffwd_layer_norm(x))
        return x
