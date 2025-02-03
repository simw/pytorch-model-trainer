import torch

from pytorch_model_trainer.data.dataloader import EpochDataLoader, RandomDataLoader


def test_epoch_dataloader() -> None:
    data = list(range(100))
    context_size = 3
    batch_size = 8
    dataloader = EpochDataLoader(data, context_size, batch_size)
    assert len(dataloader) == len(data) // (context_size * batch_size)
    all_inputs = []
    all_targets = []
    for inputs, targets in dataloader:
        assert inputs.shape == (batch_size, context_size)
        assert targets.shape == (batch_size, context_size)
        assert (inputs[:, 1:] == targets[:, :-1]).all()
        all_inputs.append(inputs.view(-1))
        all_targets.append(targets.view(-1))

    input_lst = sorted(torch.concat(all_inputs).tolist())
    target_lst = sorted(torch.concat(all_targets).tolist())
    assert len(input_lst) == len(data) - (len(data) % (context_size * batch_size))
    assert input_lst == data[: len(input_lst)]

    assert len(target_lst) == len(input_lst)
    assert target_lst == data[1 : (len(input_lst) + 1)]


def test_random_dataloader() -> None:
    data = list(range(100))
    context_size = 3
    batch_size = 8
    dataloader = RandomDataLoader(data, context_size, batch_size)
    assert len(dataloader) == len(data) // (context_size * batch_size)
    data = iter(dataloader)
    inputs, targets = next(data)
    assert inputs.shape == (batch_size, context_size)
    assert targets.shape == (batch_size, context_size)
    assert (inputs[:, 1:] == targets[:, :-1]).all()

    inputs2, targets2 = next(data)
    assert (inputs2 != inputs).any()
    assert (targets2 != targets).any()
