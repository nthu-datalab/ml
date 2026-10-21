"""PyTorch port of `labs/11-1_CNN/lab11_1_lib.py`.

The original `draw_timeline()` plotting logic is framework-agnostic (matplotlib
+ numpy); it only touched TensorFlow via `.numpy()` calls to pull data out of
TF tensors. Here it takes plain numpy arrays directly. `TimeMeasuredDataset`
and `timing_collate` are new: a map-style `torch.utils.data.Dataset` that
records the same (Open, Read, Map, ...) wall-clock timings as the original
TF `TimeMeasuredDataset`, so the same before/after-optimization benchmarking
story from the TensorFlow lab can be told with `DataLoader(num_workers=...,
pin_memory=..., prefetch_factor=...)` instead of `tf.data`'s
`interleave/cache/prefetch`.
"""
import itertools
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeMeasuredDataset(Dataset):
    _instances_counter = itertools.count()

    def __init__(self, open_file, read_file, filename=None, transform=None,
                 simulate_io_latency=0.0):
        self.instance_idx = next(TimeMeasuredDataset._instances_counter)
        self.read_file = read_file
        self.transform = transform
        self.simulate_io_latency = simulate_io_latency

        open_enter = time.perf_counter()
        self.paths, self.labels = open_file(filename)
        self._open_timing = (open_enter, time.perf_counter() - open_enter)
        self._opened_once = False
        # NOTE: with DataLoader(num_workers > 0) each worker process gets its
        # own *copy* of this Dataset (via fork), so this counter is scoped per
        # worker, not global -- fine for illustrating the timeline, but don't
        # treat epoch_idx/sample_idx below as an exact global count.
        self._call_count = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        epoch_idx, sample_idx = divmod(self._call_count, len(self))
        self._call_count += 1

        read_enter = time.perf_counter()
        if self.simulate_io_latency:
            time.sleep(self.simulate_io_latency)
        img = self.read_file(self.paths[idx])
        read_elapsed = time.perf_counter() - read_enter

        map_enter = time.perf_counter()
        if self.transform is not None:
            img = self.transform(img)
        map_elapsed = time.perf_counter() - map_enter

        if not self._opened_once:
            open_entry = self._open_timing
            self._opened_once = True
        else:
            open_entry = (-1.0, -1.0)  # negative values are filtered out by draw_timeline

        steps = ['Open', 'Read', 'Map']
        times = [open_entry, (read_enter, read_elapsed), (map_enter, map_elapsed)]
        values = [(self.instance_idx, epoch_idx, -1),
                  (self.instance_idx, epoch_idx, sample_idx),
                  (self.instance_idx, epoch_idx, sample_idx)]
        return steps, times, values, img, self.labels[idx]


def timing_collate(batch):
    """Keeps the (steps, times, values) timing records as plain per-sample
    lists instead of stacking them, while collating (img, label) normally."""
    steps, times, values, imgs, labels = zip(*batch)
    imgs = torch.utils.data.default_collate(imgs)
    labels = torch.utils.data.default_collate(labels)
    return list(steps), list(times), list(values), imgs, labels


def draw_timeline(timeline, title, min_width=0.5, annotate=False, save=False):
    steps = np.asarray(timeline['steps'])
    times = np.asarray(timeline['times'], dtype=np.float64)
    values = np.asarray(timeline['values'], dtype=np.int64)

    # Remove invalid entries (negative times, or empty steps) from the timelines
    invalid_mask = np.logical_and(times[:, 0] > 0, steps != '')
    steps = steps[invalid_mask]
    times = times[invalid_mask].copy()
    values = values[invalid_mask]

    # Get a set of different steps, ordered by the first time they are encountered
    step_ids, indices = np.unique(steps, return_index=True)
    step_ids = step_ids[np.argsort(indices)]

    # Shift the starting time to 0 and compute the maximal time value
    min_time = times[:, 0].min()
    times[:, 0] = times[:, 0] - min_time
    end = max(min_width, (times[:, 0] + times[:, 1]).max() + 0.01)

    cmap = mpl.colormaps.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    axs = np.atleast_1d(axs)
    fig.suptitle(title)
    fig.set_size_inches(17.0, len(step_ids))
    plt.xlim(-0.01, end)

    order = ['Open', 'Read', 'Map', 'Train', 'Val', 'Test', 'Epoch']
    step_ids = sorted(step_ids, key=lambda s: order.index(s) if s in order else len(order))

    for i, step in enumerate(step_ids):
        ax = axs[i]
        ax.set_ylabel(step)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        entries_mask = steps == step
        serie = times[entries_mask]
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join(f"{l}: {v}" for l, v in zip(("i", "e", "s"), annotations[j]))
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")
