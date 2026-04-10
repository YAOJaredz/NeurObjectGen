import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from HexPred.object_response.get_rust_response import get_rust_responses
# from HexPred.object_response.get_hvm_response import get_hvm_responses
from HexPred.constants import ALL_MONKEYS


def make_rust_loader(batch_size=64, seed=42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for the RUST neural response dataset.

    Loads per-area spike responses for every monkey in ``ALL_MONKEYS`` over the
    0-250 ms post-stimulus window, concatenates them along the neuron axis,
    drops neurons whose responses are entirely zero or NaN across all stimuli
    and time bins, and splits the 300 stimuli into fixed 200/50/50
    train/val/test partitions using a seeded permutation.

    Each sample yielded by the loaders is a single stimulus with shape
    ``(neurons, time)``.

    Args:
        batch_size: Number of stimuli per batch. Defaults to 64.
        seed: Seed for the NumPy RNG used to permute stimulus indices. Fixing
            this guarantees reproducible train/val/test splits. Defaults to 42.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)`` of PyTorch
        ``DataLoader`` objects. The training loader is shuffled; the
        validation and test loaders preserve stimulus order.
    """
    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(
            mode='area', area='all', monkey=monkey, time_window=(0, 250)
        )
        monkey_responses.append(rsp)

    # stimuli x neurons x time
    object_responses = np.concatenate(monkey_responses, axis=1)

    # drop neurons that are all-zero / all-NaN across stimuli and time
    dead_neuron = np.all(
        (object_responses == 0) | np.isnan(object_responses), axis=(0, 2)
    )
    object_responses = object_responses[:, ~dead_neuron, :]

    assert object_responses.shape[0] == 300, (
        f"expected 300 stimuli, got {object_responses.shape[0]}"
    )

    tensor = torch.from_numpy(object_responses).float()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(300)
    train_idx = torch.from_numpy(perm[:200]).long()
    val_idx = torch.from_numpy(perm[200:250]).long()
    test_idx = torch.from_numpy(perm[250:]).long()

    train_set = TensorDataset(tensor[train_idx])
    val_set = TensorDataset(tensor[val_idx])
    test_set = TensorDataset(tensor[test_idx])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(
        f"RUST loaders created: "
        f"train={train_idx.shape[0]}, "
        f"val={val_idx.shape[0]}, "
        f"test={test_idx.shape[0]}, "
        f"neurons={object_responses.shape[1]}, "
        f"time={object_responses.shape[2]}"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    make_rust_loader()