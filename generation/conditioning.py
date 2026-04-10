"""Compose conditioning signals: neural-only, text-only, neural+text."""


def neural_only(neural_embedding):
    raise NotImplementedError


def text_only(caption: str):
    raise NotImplementedError


def neural_plus_text(neural_embedding, caption: str):
    raise NotImplementedError
