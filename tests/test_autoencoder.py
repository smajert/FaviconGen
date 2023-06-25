import random

import torch

import logo_maker.autoencoder as testee


def test_autoencoder_model_runs(device: str = "cuda"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    pseudo_labels = torch.randint(0, 9, size=(32,), device=device)
    model = testee.AutoEncoder(3, 32, 10).to(device)
    test_output = model(pseudo_batch, pseudo_labels)[0]
    if device == "cpu":
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(-0.1552, device=device), rtol=0, atol=1e-4)
    elif device == "cuda":
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(0.1188, device=device), rtol=0, atol=1e-4)


def test_patch_discriminator_model_run(device: str = "cpu"):
    torch.random.manual_seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    model = testee.PatchDiscriminator(3).to(device)
    test_output = model(pseudo_batch)


