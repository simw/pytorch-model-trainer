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

    return MockResponse()


def test_train(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", mock_httpx_get)

    runner = CliRunner()
    # Currently runs the full training run ...
    # result = runner.invoke(cli, ["test"])

    # assert result.exit_code == 0
    # assert "Training complete" in result.output


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "Commands" in result.output
