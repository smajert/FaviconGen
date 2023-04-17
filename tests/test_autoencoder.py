import random

import torch

import logo_maker.autoencoder as auto


def test_autoencoder_model_runs(device: str = "cpu"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    model = auto.AutoEncoder().to(device)
    test_output = model(pseudo_batch)
    print(torch.mean(test_output))
    # if device == "cpu":
    #     torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(-0.0508, device=device), rtol=0, atol=1e-4)
    # elif device == "cuda":
    #     torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(-0.0885, device=device), rtol=0, atol=1e-4)
    # else:
    #     pass


def test_patch_discriminator_model_rund(device: str = "cpu"):
    torch.random.manual_seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    model = auto.PatchDiscriminator().to(device)
    test_output = model(pseudo_batch)
    print(test_output.shape)


