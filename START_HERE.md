# Thesis: VS Code and PyTorch

This folder is the working project for the bearing fault-detection thesis.
VS Code is the editor. Python runs our scripts. PyTorch supplies neural networks
and GPU computation. The `.venv` directory holds the project's own Python packages.

## Verified setup

Verified on 19 October 2026:

- Python 3.12.3 in this project's `.venv`.
- PyTorch 2.13.0+cu126 and torchvision 0.28.0+cu126.
- CUDA 12.6 runtime with `torch.cuda.is_available()` returning `True`.
- NVIDIA GeForce RTX 3060 Laptop GPU, approximately 6 GiB memory.
- A synthetic GPU Conv1D forward/backward/optimizer step completed successfully.
- MATLAB read/write, plotting, metrics, and TensorBoard checks passed.
- `pip check` found no broken requirements.

The machine-readable result is `outputs/setup/setup_check.json`. The complete
resolved package list is `requirements-lock.txt`.

## Open and run

1. Open this folder in VS Code: `C:\Users\TUF Gaming\Documents\Thesis`.
2. Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select
   `.venv\Scripts\python.exe` (Python 3.12). Use **Enter interpreter path** if needed.
3. Open **Terminal > New Terminal**.
4. Run the following command in that terminal:

```powershell
.\.venv\Scripts\python.exe scripts\check_setup.py --require-cuda
```

The explicit interpreter path works even if PowerShell environment activation
is disabled. No execution-policy change is needed.

Alternatively, open `scripts/check_setup.py`, choose the **Thesis: check PyTorch
and GPU** debug configuration in **Run and Debug**, and press `F5`.

## What success means

The script should show `CUDA available: True`, identify the RTX 3060, perform
a small one-dimensional CNN training step, and print `SETUP CHECK PASSED`.
It also writes a JSON report, an example MATLAB file, a plot, and a TensorBoard
event under `outputs/setup/`.

All inputs in this check are synthetic. Its loss is not a bearing-diagnosis result.
No CWRU dataset or research model has been trained by this setup script.

## Recreate the environment

Run from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_setup.py --require-cuda
```

The PyTorch pair is pinned to 2.13.0 / torchvision 0.28.0 with CUDA 12.6 using
the official wheel index. This is a documented version pair, not a claim to be
the newest release. The existing NVIDIA driver must support that runtime.

For the exact package versions verified in this setup, after creating `.venv`
and upgrading pip, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The initial GPU-package download stalled in pip. The setup used a direct
download from the official PyTorch server, verified its SHA-256 checksum, and
installed that wheel. Its checksum was:
`380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2`.
The downloaded wheel remains inside `.venv` and is ignored by Git.

Sources: [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
and [local installation and verification](https://pytorch.org/get-started/locally/).

## Next milestone

Load a small set of official CWRU recordings, verify their channel and sampling
rate metadata, and plot healthy and faulty vibration signals. Then define the
source/target split before training the first source-only classifier.
