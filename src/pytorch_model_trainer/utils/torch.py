import torch


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_torch_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def get_torch_devices() -> list[str]:
    return (
        ["cpu"]
        + (["cuda"] if torch.cuda.is_available() else [])
        + (["mps"] if torch.backends.mps.is_available() else [])
    )
