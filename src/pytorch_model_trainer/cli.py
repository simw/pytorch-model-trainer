import logging

import click

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
    settings = {
        "data": {
            "url": "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt",
            "encoder": "gpt2",
            "train_test_split": 0.9,
        },
        "model": {
            "max_context_size": 128,
            "n_embeddings": 32 * 4,
            "num_attention_heads": 4,
            "feed_forward_mult": 4,
            "p_dropout": 0.2,
            "ff_nonlinearity": "gelu",
            "num_attention_blocks": 4,
        },
        "optimization": {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "epochs": 50,
        },
        "reporting": {
            "reports_per_epoch": 10,
            "testing_batches": 20,
        },
        "seed": 123,
    }
    parsed_settings = TrainingSettings(**settings)
    train(parsed_settings)
