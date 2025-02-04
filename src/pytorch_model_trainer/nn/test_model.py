import torch


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._linear1 = torch.nn.Linear(1, 10)
        self._linear2 = torch.nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = torch.relu(x)
        x = self._linear2(x)
        return x
