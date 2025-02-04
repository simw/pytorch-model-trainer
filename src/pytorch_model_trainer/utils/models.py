from collections.abc import Callable, Iterator
from typing import Protocol

import torch


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...  # pragma: no cover

    def decode(self, tokens: list[int]) -> str: ...  # pragma: no cover


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: str,
    encoder: Encoder,
    max_new_tokens: int,
    n_samples: int,
) -> list[str]:
    torch_device = next(model.parameters()).device
    tokens = [encoder.encode(prompt)] * n_samples
    values = torch.tensor(tokens).to(torch_device)

    for _ in range(max_new_tokens):
        logits = model(values)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_value = torch.multinomial(probs, num_samples=1)
        values = torch.cat((values, next_value), dim=1)

    return [encoder.decode(sample.tolist()) for sample in values]


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    n_batches: int,
) -> float:
    model.eval()
    losses = torch.zeros(n_batches)
    torch_device = next(model.parameters()).device
    for k in range(n_batches):
        inputs, targets = next(dataloader)
        logits = model(inputs.to(torch_device))
        loss = loss_fn(logits, targets.to(torch_device))
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


def count_params(model: torch.nn.Module, depth: int = 0) -> dict[str, int]:
    params = {name: param.numel() for name, param in model.named_parameters()}

    result: dict[str, int] = {}
    for name, param in params.items():
        adjusted_name = ".".join(name.split(".")[:depth])
        result[adjusted_name] = result.get(adjusted_name, 0) + param

    return result
