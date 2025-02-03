import pytest
import torch

from pytorch_model_trainer.nn.model import Model, ModelSettings
from pytorch_model_trainer.utils.torch import get_torch_devices


@pytest.mark.parametrize("torch_device", get_torch_devices())
def test_model(torch_device: str) -> None:
    vocab_size = 11
    settings = ModelSettings(
        n_embeddings=9,
        max_context_size=7,
        num_attention_heads=3,
        feed_forward_mult=4,
        p_dropout=0.15,
        ff_nonlinearity="relu",
        num_attention_blocks=3,
    )

    model = Model(vocab_size, settings).to(torch_device)

    batch_size = 32
    context_size = 4
    x = torch.ones((batch_size, context_size), dtype=torch.long, device=torch_device)
    logits = model(x)
    assert logits.shape == (batch_size, context_size, vocab_size)
