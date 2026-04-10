"""Reconstruction metrics: cosine similarity, SSIM, 2AFC identification."""


def cosine_similarity(pred, target):
    raise NotImplementedError


def ssim(pred_image, target_image):
    raise NotImplementedError


def two_afc_identification(pred_embeddings, target_embeddings):
    raise NotImplementedError
