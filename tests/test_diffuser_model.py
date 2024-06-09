import torch

from favicon_gen import diffuser_model
from favicon_gen.blocks import VarianceSchedule


def test_saving_and_loading_of_model_runs(tmp_path):
    schedule = VarianceSchedule((0.1, 0.001), 20, device="cpu")
    model = diffuser_model.DiffusersModel(1, schedule, 10, 2)
    out_file = tmp_path / "model.pt"
    torch.save(model.state_dict(), out_file)

    assert out_file.exists()
    loader = diffuser_model.DiffusersModel(1, schedule, 10, 2)
    loader.load_state_dict(torch.load(out_file))


def test_getting_device_works():
    schedule = VarianceSchedule((0.1, 0.001), 20, device="cpu")
    model = diffuser_model.DiffusersModel(1, schedule, 10, 20)
    assert model.device == torch.device("cpu")
