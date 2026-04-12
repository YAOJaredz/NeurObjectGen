import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from HexPred.object_response.get_rust_response import get_rust_responses
# from HexPred.object_response.get_hvm_response import get_hvm_responses
from HexPred.constants import ALL_MONKEYS

from config_const import N_STIMULI, N_TRAIN, N_VAL, RUST_TIME_WINDOW, SEED, SIGLIP_EMBEDDINGS_PATH
from data_utils.stimuli import load_rust_stimuli


def make_rust_loader(
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = True,
    use_embeddings: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for neural-response → image reconstruction.

    Loads per-area spike responses for every monkey in ``ALL_MONKEYS`` over the
    0-250 ms post-stimulus window, concatenates them along the neuron axis,
    drops neurons whose responses are entirely zero or NaN across all stimuli
    and time bins, and splits the 300 stimuli into fixed 200/50/50
    train/val/test partitions using a seeded permutation.

    Each sample yielded by the loaders is ``(neural_response, target)`` where
    ``neural_response`` has shape ``(neurons, time)`` and ``target`` is either
    an image of shape ``(3, 224, 224)`` or a pre-cached SigLIP embedding of
    shape ``(D,)`` depending on ``use_embeddings``.

    Args:
        batch_size:      Number of stimuli per batch. Defaults to 64.
        seed:            Seed for the NumPy RNG used to permute stimulus indices.
        verbose:         Print loader statistics.
        use_embeddings:  If True, yield pre-cached SigLIP embeddings as targets
                         instead of raw images. Requires the cache at
                         SIGLIP_EMBEDDINGS_PATH (run scripts/cache_siglip.py).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)`` of PyTorch
        ``DataLoader`` objects. The training loader is shuffled; the
        validation and test loaders preserve stimulus order.
    """
    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(
            mode='area', area='all', monkey=monkey, time_window=RUST_TIME_WINDOW
        )
        monkey_responses.append(rsp)

    # stimuli x neurons x time
    object_responses = np.concatenate(monkey_responses, axis=1)

    # drop neurons that are all-zero or contain any NaN across stimuli and time
    dead_neuron = (
        np.all((object_responses == 0) | np.isnan(object_responses), axis=(0, 2))
        | np.any(np.isnan(object_responses), axis=(0, 2))
    )
    object_responses = object_responses[:, ~dead_neuron, :]
    # replace any remaining zeros with zero (no-op) and verify clean
    assert not np.isnan(object_responses).any(), "NaN values remain after neuron filtering."

    assert object_responses.shape[0] == N_STIMULI, (
        f"expected {N_STIMULI} stimuli, got {object_responses.shape[0]}"
    )

    neural_tensor = torch.from_numpy(object_responses).float()  # (N_STIMULI, neurons, time)

    if use_embeddings:
        if not SIGLIP_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"SigLIP cache not found at {SIGLIP_EMBEDDINGS_PATH}. "
                "Run: python scripts/cache_siglip.py"
            )
        target_tensor = torch.load(SIGLIP_EMBEDDINGS_PATH, weights_only=True)  # (N_STIMULI, D)
    else:
        target_tensor = load_rust_stimuli()  # (N_STIMULI, 3, 224, 224)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
    test_idx = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

    def make(idx, shuffle):
        return DataLoader(
            TensorDataset(neural_tensor[idx], target_tensor[idx]),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    if verbose:
        target_shape = tuple(target_tensor.shape[1:])
        print(
            f"RUST loaders created: "
            f"train={train_idx.shape[0]}, "
            f"val={val_idx.shape[0]}, "
            f"test={test_idx.shape[0]}, "
            f"neurons={object_responses.shape[1]}, "
            f"time={object_responses.shape[2]}, "
            f"target={target_shape}"
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    make_rust_loader()