import pytest
import torch

from pytorch_model_trainer.utils.torch import (
    get_torch_device,
    get_torch_devices,
    set_torch_seed,
)


def test_set_torch_seed() -> None:
    seed = 42
    set_torch_seed(seed)

    assert torch.initial_seed() == seed
    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == seed
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_get_torch_device_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_torch_device() == "cpu"


def test_get_torch_device_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_torch_device() == "cuda"


def test_get_torch_device_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_torch_device() == "mps"


def test_get_torch_device_with_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate no CUDA and no MPS available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_torch_device() == "cpu"
    assert get_torch_devices() == ["cpu"]

    # Simulate CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_torch_device() == "cuda"
    assert "cuda" in get_torch_devices()

    # Simulate MPS available
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_torch_device() == "cuda"  # CUDA takes precedence if both are available
    assert "mps" in get_torch_devices()
