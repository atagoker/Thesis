"""Check the thesis environment with a synthetic 1D CNN training step.

The random inputs verify software operation; they are not bearing data and
the printed loss is not an experimental thesis result.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import sklearn.metrics
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Interpreter: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA runtime: {torch.version.cuda}", flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}", flush=True)
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but unavailable. Check the selected interpreter and PyTorch build.")

    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"Device: {gpu_name or 'CPU'}", flush=True)
    torch.manual_seed(42)

    # Check a real CUDA operation against a CPU reference.
    left = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
    right = torch.eye(16, dtype=torch.float32)
    actual = (left.to(device) @ right.to(device)).cpu()
    torch.testing.assert_close(actual, left @ right)

    model = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 4),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(8, 1, 2048, device=device)
    labels = torch.arange(8, device=device) % 4
    before = model[-1].weight.detach().clone()
    logits = model(inputs)
    assert logits.shape == (8, 4), logits.shape
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")
    optimizer.zero_grad()
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or invalid gradient: {name}")
    optimizer.step()
    if torch.equal(before, model[-1].weight.detach()):
        raise RuntimeError("The optimizer did not update the classifier weights.")
    if cuda_available:
        torch.cuda.synchronize()

    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "setup"
    output.mkdir(parents=True, exist_ok=True)

    # Also exercise the scientific tools needed for the next CWRU session.
    sample_rate = 12000
    time = np.arange(2048) / sample_rate
    signal = np.sin(2 * np.pi * 100 * time)
    mat_path = output / "synthetic_signal.mat"
    scipy.io.savemat(mat_path, {"synthetic_signal": signal})
    reloaded = scipy.io.loadmat(mat_path)["synthetic_signal"].reshape(-1)
    np.testing.assert_allclose(reloaded, signal)
    assert sklearn.metrics.accuracy_score([0, 1], [0, 1]) == 1.0
    assert len(pd.DataFrame({"signal": signal})) == 2048

    figure, axis = plt.subplots(figsize=(9, 3))
    axis.plot(time, signal)
    axis.set(xlabel="Time (s)", ylabel="Synthetic amplitude", title="Software check: synthetic 100 Hz sine, not CWRU data")
    figure.tight_layout()
    figure.savefig(output / "synthetic_signal.png", dpi=150)
    plt.close(figure)
    with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
        writer.add_scalar("software_check/synthetic_loss", loss.item(), 0)

    package_names = ["torch", "torchvision", "numpy", "scipy", "matplotlib", "pandas", "scikit-learn", "tqdm", "tensorboard"]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Software smoke test with synthetic inputs, not a bearing experiment",
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda_available else None,
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "synthetic_loss": loss.item(),
        "checks": ["matrix multiplication", "Conv1d forward", "finite backward gradients", "optimizer update", "MATLAB round trip", "plot export", "metrics", "TensorBoard event"],
        "status": "passed",
    }
    (output / "setup_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic training step passed. Loss: {loss.item():.6f}", flush=True)
    print(f"Report: {output / 'setup_check.json'}", flush=True)
    print("SETUP CHECK PASSED. This was a software check, not a fault-detection result.", flush=True)


if __name__ == "__main__":
    main()
