import math

import torch

from pytorch_model_trainer.utils.models import count_params, estimate_loss, generate


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 20)
        self.layer2 = torch.nn.Linear(20, 30)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_size = x.shape
        vocab_size = 10  # Assume a vocabulary size of 10 for simplicity
        return torch.ones((batch_size, context_size, vocab_size))


class MockEncoder:
    def encode(self, input_str: str) -> list[int]:
        return [ord(c) for c in input_str]

    def decode(self, token_list: list[int]) -> str:
        return "".join(chr(token) for token in token_list)


def test_generate() -> None:
    model = MockModel()
    encoder = MockEncoder()
    input_str = "test"
    max_new_tokens = 5
    n_samples = 2

    output = generate(model, input_str, encoder, max_new_tokens, n_samples)

    assert len(output) == n_samples
    for sample in output:
        assert len(sample) == len(input_str) + max_new_tokens
        assert sample.startswith(input_str)


def test_estimate_loss() -> None:
    def cross_entropy_loss(
        results: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        vocab_size = results.shape[-1]
        return torch.nn.functional.cross_entropy(
            results.view(-1, vocab_size), targets.view(-1)
        )

    model = MockModel()
    inputs, targets = torch.ones((32, 4), dtype=torch.long), torch.ones(
        (32, 4), dtype=torch.long
    )
    dataloader = iter([(inputs, targets)])

    mean_loss = estimate_loss(model, cross_entropy_loss, dataloader, n_batches=1)
    # Cross-entropy loss for a uniform distribution with 10 classes
    assert math.isclose(mean_loss, -math.log(0.1), abs_tol=1e-6)


def test_count_params() -> None:
    model = MockModel()
    params = count_params(model, depth=1)

    assert params == {
        "layer1": 10 * 20 + 20,
        "layer2": 20 * 30 + 30,
    }
