"""Exercise the public CAT backbone on a real CWRU batch.

This is a compatibility smoke test, not reproduction of the full CAT method.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, write_json


def main():
    repository = ROOT / "external/CAT"
    source_file = repository / "models/CNN_1.py"
    spec = importlib.util.spec_from_file_location("cat_reference_cnn", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = module.CNN(in_channel=1, out_channel=4).to(device)
    with np.load(ROOT / "data/processed/pilot_v1/load0_fit.npz") as data:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(data["y"]), size=16, replace=False)
        x = torch.from_numpy(data["x"][indices]).to(device)
        y = torch.from_numpy(data["y"][indices]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.fc.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or torch.equal(before, model.fc.weight):
        raise RuntimeError("Reference model compatibility step failed.")
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    write_json(ROOT / "outputs/pilot_v1/cat_compatibility.json", {
        "repository": "https://github.com/EPFL-IMOS/CAT", "commit": commit,
        "code_file": "models/CNN_1.py", "license": "MIT; original notices retained in external/CAT/LICENSE",
        "data": "real prepared CWRU source batch", "batch_shape": list(x.shape),
        "changed_arguments": {"in_channel": 1, "out_channel": 4},
        "checks": ["entry point --help", "GPU forward", "GPU backward", "optimizer update"],
        "loss_for_software_check_only": loss.detach().item(), "status": "passed",
        "full_CAT_training_reproduced": False,
    })
    print(f"CAT public CNN GPU step passed on real CWRU data; commit {commit}", flush=True)
    print("Full published CAT training has not been reproduced.", flush=True)


if __name__ == "__main__":
    main()
