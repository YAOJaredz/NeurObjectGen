"""FLUX.1-dev + IP-Adapter pipeline for image reconstruction."""


def load_pipeline(device: str = "cuda"):
    raise NotImplementedError


def generate(pipeline, image_embedding, prompt: str | None = None):
    raise NotImplementedError
