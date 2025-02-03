from collections.abc import Iterator

import torch


class EpochDataLoader:
    def __init__(self, tokens: list[int], context_size: int, batch_size: int) -> None:
        self._context_size = context_size
        self._batch_size = batch_size

        full_length = len(tokens)
        tokens_per_batch = context_size * batch_size
        num_batches = (full_length - 1) // tokens_per_batch
        batched_length = tokens_per_batch * num_batches
        self._used_tokens = torch.tensor(tokens[: (batched_length + 1)])

        starting_indices = context_size * torch.randperm(num_batches * batch_size).view(
            -1, batch_size, 1
        )
        self._indices = starting_indices + torch.arange(context_size).view(
            1, 1, context_size
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for ixs in self._indices:
            inputs = torch.gather(self._used_tokens, 0, ixs.view(-1)).view(
                self._batch_size, self._context_size
            )
            targets = torch.gather(self._used_tokens, 0, ixs.view(-1) + 1).view(
                self._batch_size, self._context_size
            )
            yield inputs, targets


class RandomDataLoader:
    def __init__(self, tokens: list[int], context_size: int, batch_size: int) -> None:
        self._data = torch.tensor(tokens)
        self._context_size = context_size
        self._batch_size = batch_size

    def __len__(self) -> int:
        return len(self._data) // (self._context_size * self._batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # returns tuple of 2D tensors, (batch_size, context_size)
        starting_indices = torch.randint(
            len(self._data) - (self._context_size * self._batch_size + 1),
            (self._batch_size,),
        )
        indices = torch.arange(self._context_size).unsqueeze(
            0
        ) + starting_indices.unsqueeze(1)
        inputs = torch.gather(self._data, 0, indices.view(-1)).view(
            self._batch_size, self._context_size
        )
        targets = torch.gather(self._data, 0, indices.view(-1) + 1).view(
            self._batch_size, self._context_size
        )
        return inputs, targets
