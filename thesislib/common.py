"""Shared paths, serialization, model, and evaluation-independent utilities."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["healthy", "inner_race", "rolling_element", "outer_race"]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fail on unsupported deterministic operations rather than silently vary.
    torch.use_deterministic_algorithms(True)


class BearingCNN(nn.Module):
    """Small custom 1D CNN. This is not the published CAT architecture."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        # GroupNorm avoids dependence on target batch statistics in this pilot.
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 31, stride=4, padding=15),
            nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9, stride=2, padding=4),
            nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def normalize_windows(x: np.ndarray) -> np.ndarray:
    """Per-window z score: label-free; suppresses absolute amplitude clues."""
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean(axis=-1, keepdims=True)) /
            np.maximum(x.std(axis=-1, keepdims=True), 1e-6)).astype(np.float32)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, alpha: float) -> None:
    for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
        t.mul_(alpha).add_(s, alpha=1 - alpha)
    for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
        t.copy_(s)


def masked_pseudo_loss(logits, pseudo_labels, mask):
    # Full-batch denominator: low coverage reduces the target contribution.
    losses = nn.functional.cross_entropy(logits, pseudo_labels, reduction="none")
    return (losses * mask.float()).mean()
