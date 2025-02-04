import logging

import click
import torch

from pytorch_model_trainer.nn.test_model import TestModel
from pytorch_model_trainer.utils.torch import get_torch_device, set_torch_seed

from .train import TrainingSettings, train


@click.group()
@click.version_option()
def cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    )


@cli.command()
def test() -> None:
    logger = logging.getLogger(__name__)

    torch_device = get_torch_device()
    logger.info(f"Using device: {torch_device}")
    set_torch_seed(123)

    model = TestModel().to(torch_device)
    logger.info("Compiling model")
    model = torch.compile(model)  # type: ignore

    x = torch.randn(1, 1)
    result = model(x)
    logger.info(f"Result: {result}")
    assert result.shape == (1, 1)  # noqa: S101


@cli.command()
@click.argument("settings_json_string")
def v1(settings_json_string: str) -> None:
    parsed_settings = TrainingSettings.model_validate_json(settings_json_string)
    train(parsed_settings)
