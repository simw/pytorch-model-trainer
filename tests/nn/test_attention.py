import pytest
import torch

from pytorch_model_trainer.nn.attention import (
    AttentionBlock,
    AttentionError,
    FeedForward,
    MultiHeadAttention,
)
from pytorch_model_trainer.utils.torch import get_torch_devices

torch_devices = get_torch_devices()


@pytest.mark.parametrize("torch_device", torch_devices)
def test_multi_head_attention(torch_device: str) -> None:
    max_context_size = 7
    input_dim = 11
    head_size = 5
    num_heads = 4
    p_dropout = 0.2
    p_multihead_dropout = 0.15

    heads = MultiHeadAttention(
        max_context_size,
        input_dim,
        head_size,
        p_dropout,
        num_heads,
        p_multihead_dropout,
    ).to(torch_device)

    batch_size = 32
    context_size = 4
    x = torch.ones((batch_size, context_size, input_dim), device=torch_device)
    result = heads(x)
    assert result.shape == (batch_size, context_size, head_size * num_heads)
    assert result.device.type == torch_device


@pytest.mark.parametrize("torch_device", torch_devices)
def test_feed_forward(torch_device: str) -> None:
    n_input = 5
    n_hidden = 9
    n_output = 7
    p_dropout = 0.2
    layer = FeedForward(n_input, n_hidden, "relu", n_output, p_dropout).to(torch_device)

    batch_size = 11
    other_size = 13
    inputs = torch.ones((batch_size, other_size, n_input), device=torch_device)
    result = layer(inputs)
    assert result.shape == (batch_size, other_size, n_output)

    layer = FeedForward(n_input, n_hidden, "gelu", n_output, p_dropout).to(torch_device)
    result = layer(inputs)
    assert result.shape == (batch_size, other_size, n_output)
    assert result.device.type == torch_device


@pytest.mark.parametrize("torch_device", torch_devices)
def test_attention_block(torch_device: str) -> None:
    input_dim = 12

    block = AttentionBlock(
        max_context_size=7,
        input_dim=input_dim,
        p_attention_dropout=0.2,
        num_attention_heads=4,
        p_multihead_attention_dropout=0.15,
        feed_forward_mult=3,
        ff_nonlinearity="relu",
        p_ffwd_dropout=0.1,
    ).to(torch_device)

    batch_size = 32
    context_size = 4
    x = torch.ones((batch_size, context_size, input_dim), device=torch_device)
    result = block(x)
    assert result.shape == (batch_size, context_size, input_dim)
    assert result.device.type == torch_device

    with pytest.raises(AttentionError):
        block = AttentionBlock(
            max_context_size=7,
            input_dim=13,  # Input dim not divisible by num_heads
            p_attention_dropout=0.2,
            num_attention_heads=4,
            p_multihead_attention_dropout=0.15,
            feed_forward_mult=3,
            ff_nonlinearity="relu",
            p_ffwd_dropout=0.1,
        )
