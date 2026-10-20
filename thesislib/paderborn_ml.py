"""Label-agnostic probability inference and recording aggregation."""

import numpy as np
import torch


def record_ids(window_ids):
    return np.asarray([str(identifier).rsplit(":", 1)[0] for identifier in window_ids])


def aggregate_recordings(probabilities, window_ids, labels=None):
    records, inverse = np.unique(record_ids(window_ids), return_inverse=True)
    summed = np.zeros((len(records), probabilities.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, probabilities)
    means = summed / np.bincount(inverse)[:, None]
    if labels is None:
        return records, means, None
    grouped_labels = np.asarray([labels[np.flatnonzero(inverse == i)[0]] for i in range(len(records))])
    if not np.array_equal(grouped_labels[inverse], labels):
        raise ValueError("Conflicting labels within a recording.")
    return records, means, grouped_labels


@torch.no_grad()
def probabilities(model, x, device, batch_size=256):
    model.eval()
    result = []
    for batch in x.split(batch_size):
        result.append(model(batch.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(result)
