import logging
import time

import httpx
import tiktoken
import torch
from pydantic import BaseModel

from pytorch_model_trainer.data.dataloader import EpochDataLoader, RandomDataLoader
from pytorch_model_trainer.nn.model import Model, ModelSettings
from pytorch_model_trainer.utils.models import estimate_loss, generate
from pytorch_model_trainer.utils.torch import get_torch_device, set_torch_seed

logger = logging.getLogger(__name__)


class DataSettings(BaseModel):
    url: str
    encoder: str
    train_test_split: float


class OptimizationSettings(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int


class ReportingSettings(BaseModel):
    reports_per_epoch: int
    testing_batches: int


class TrainingSettings(BaseModel):
    data: DataSettings
    model: ModelSettings
    optimization: OptimizationSettings
    reporting: ReportingSettings
    seed: int


def _cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = inputs.shape[-1]
    return torch.nn.functional.cross_entropy(
        inputs.view(-1, vocab_size), targets.view(-1)
    )


def train(settings: TrainingSettings) -> None:
    set_torch_seed(settings.seed)

    # Model
    torch_device = get_torch_device()
    logger.info(f"Using device: {torch_device}")
    tokenizer = tiktoken.get_encoding(settings.data.encoder)
    model = Model(tokenizer.n_vocab, settings.model)
    model = model.to(torch_device)
    if torch_device == "cuda":
        model = torch.compile(model)  # type: ignore

    # Data
    full_text = httpx.get(settings.data.url).text
    logger.info(f"Loaded text with {len(full_text)} characters")
    n_split = int(settings.data.train_test_split * len(full_text))
    train_tokens = tokenizer.encode(full_text[:n_split])
    test_tokens = tokenizer.encode(full_text[n_split:])
    logger.info(
        f"Training on {len(train_tokens)} tokens, testing on {len(test_tokens)} tokens"
    )
    max_context_size = settings.model.max_context_size
    batch_size = settings.optimization.batch_size
    training_dataloader = EpochDataLoader(train_tokens, max_context_size, batch_size)
    testing_dataloader = RandomDataLoader(test_tokens, max_context_size, batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.optimization.learning_rate
    )
    loss_fn = _cross_entropy_loss

    # Running the optimization
    steps_per_epoch = len(training_dataloader)
    msg = (
        f"Running {settings.optimization.epochs} epochs "
        f"each with {steps_per_epoch} steps"
    )
    logger.info(msg)
    start_time = time.monotonic()
    n_epochs = settings.optimization.epochs
    reporting_interval = steps_per_epoch // settings.reporting.reports_per_epoch
    testing_batches = settings.reporting.testing_batches

    for epoch in range(n_epochs):
        msg = (
            f"----- Epoch {epoch + 1} --- (lr = "
            f"{optimizer.param_groups[0]['lr']:.3g}) -----"
        )
        logger.info(msg)
        for step, (inputs, targets) in enumerate(training_dataloader):
            optimizer.zero_grad()
            with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
                logits = model(inputs.to(torch_device))
                loss = loss_fn(logits, targets.to(torch_device))
            loss.backward()  # type: ignore
            optimizer.step()

            if (step + 1) % reporting_interval == 0:
                test_loss = estimate_loss(
                    model, loss_fn, testing_dataloader, testing_batches
                )

                now_time = time.monotonic()
                time_diff = now_time - start_time
                token_rate = inputs.numel() * reporting_interval / time_diff
                start_time = now_time

                remaining_time = (
                    (
                        steps_per_epoch
                        - step
                        - 1
                        + (n_epochs - epoch - 1) * steps_per_epoch
                    )
                    * time_diff
                    / reporting_interval
                )
                full_step = epoch * steps_per_epoch + step
                msg = (
                    f"Step {full_step + 1}: training_loss={loss.item():.3f}, "
                    f"test_loss={test_loss:.3f}, "
                    f"time_diff={time_diff:.3g}, "
                    f"token_rate={token_rate:.3g}/s "
                    f"(remaining={remaining_time:.1f}s)"
                )
                logger.info(msg)

    # Final logging
    test_loss = estimate_loss(model, loss_fn, testing_dataloader, testing_batches)
    logger.info(f"Final: training_loss={loss.item():.3f}, test_loss={test_loss:.3f}")
    samples = generate(model, "\n", tokenizer, 100, 3)
    for i, sample in enumerate(samples):
        logging.info(f"Sample {i + 1}: {repr(sample)}")
