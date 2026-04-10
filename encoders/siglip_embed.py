"""Frozen SigLIP image encoder: defines the target space for neural decoding."""


def load_siglip(device: str = "cuda"):
    raise NotImplementedError


def embed_images(model, images):
    raise NotImplementedError
