import httpx
import pytest

from pytorch_model_trainer.train import (
    DataSettings,
    ModelSettings,
    OptimizationSettings,
    ReportingSettings,
    TrainingSettings,
    train,
)

TEXT = """
This is some fake data for the test.
But it needs to be on for a little bit so that
the train / test split can have enough data
in each part to work with.
This is some fake data for the test.
But it needs to be on for a little bit so that
the train / test split can have enough data
in each part to work with.
This is some fake data for the test.
But it needs to be on for a little bit so that
the train / test split can have enough data
in each part to work with.
"""


def mock_httpx_get(_url: str) -> httpx.Response:
    class MockResponse:
        @property
        def text(self) -> str:
            return TEXT

    return MockResponse()


def test_train(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", mock_httpx_get)

    settings = TrainingSettings(
        data=DataSettings(
            url="http://example.com",
            encoder="gpt2",
            train_test_split=0.6,
        ),
        model=ModelSettings(
            max_context_size=8,
            n_embeddings=2 * 4,
            num_attention_heads=1,
            feed_forward_mult=2,
            p_dropout=0.2,
            ff_nonlinearity="gelu",
            num_attention_blocks=2,
        ),
        optimization=OptimizationSettings(
            batch_size=2,
            learning_rate=1e-4,
            epochs=1,
        ),
        reporting=ReportingSettings(
            reports_per_epoch=2,
            testing_batches=2,
        ),
        seed=123,
    )

    train(settings)
