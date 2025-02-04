import json
from typing import cast

import httpx
import pytest
from click.testing import CliRunner

from pytorch_model_trainer.cli import cli

TEXT = """
This is some fake data for the test.
But it needs to be on for a little bit so that
the train / test split can have enough data
in each part to work with.
"""


def mock_httpx_get(_url: str) -> httpx.Response:
    class MockResponse:
        @property
        def text(self) -> str:
            return "\n".join([TEXT] * 10000)

    return cast(httpx.Response, MockResponse())


def test_train(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", mock_httpx_get)
    runner = CliRunner()

    settings = {
        "data": {
            "url": "http://example.com",
            "encoder": "gpt2",
            "train_test_split": 0.6,
        },
        "model": {
            "max_context_size": 8,
            "n_embeddings": 8,
            "num_attention_heads": 2,
            "feed_forward_mult": 2,
            "p_dropout": 0.2,
            "ff_nonlinearity": "relu",
            "num_attention_blocks": 1,
        },
        "optimization": {
            "batch_size": 4,
            "learning_rate": 3e-4,
            "epochs": 2,
        },
        "reporting": {
            "reports_per_epoch": 10,
            "testing_batches": 2,
        },
        "seed": 123,
    }

    result = runner.invoke(cli, ["v1", json.dumps(settings)])
    assert result.exit_code == 0


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "Commands" in result.output
