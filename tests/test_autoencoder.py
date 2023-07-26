import random

import torch

import favicon_gen.autoencoder as ate


def test_gpu_available():
    assert torch.cuda.is_available()


def test_autoencoder_model_runs(device: str = "cpu"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    pseudo_labels = torch.randint(0, 9, size=(32,), device=device)
    model = ate.VariationalAutoEncoder(3, 32).to(device)
    _ = model(pseudo_batch, pseudo_labels)[0]


def test_patch_discriminator_model_run(device: str = "cpu"):
    torch.random.manual_seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    model = ate.PatchDiscriminator(3).to(device)
    _ = model(pseudo_batch)
