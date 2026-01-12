from torch.utils.data import Sampler
import numpy as np
import random

class EventBalancedBatchSampler(Sampler):
    """
    Balanced batch sampler for survival analysis.

    Ensures each batch contains ~50% event and ~50% no-event samples
    (or as balanced as possible given dataset imbalance).
    """

    def __init__(self, events, batch_size, drop_last=False, seed=42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.events = np.array(events)

        # Separate event / no-event indices
        self.event_indices = np.where(self.events == 1)[0].tolist()
        self.no_event_indices = np.where(self.events == 0)[0].tolist()

        if len(self.event_indices) == 0:
            raise ValueError("No event samples found to create balanced batches.")
        if len(self.no_event_indices) == 0:
            raise ValueError("No censored samples found to create balanced batches.")

        self.rng = random.Random(seed)

    def __iter__(self):
        # Shuffle each epoch
        event_indices = self.event_indices.copy()
        no_event_indices = self.no_event_indices.copy()

        self.rng.shuffle(event_indices)
        self.rng.shuffle(no_event_indices)

        all_batches = []

        half_bs = self.batch_size // 2  # target per class
        while event_indices and no_event_indices:
            batch = []

            # Take half events
            for _ in range(min(half_bs, len(event_indices))):
                batch.append(event_indices.pop())

            # Take half no-events
            for _ in range(min(half_bs, len(no_event_indices))):
                batch.append(no_event_indices.pop())

            # If batch not full, top up with whichever class remains
            while len(batch) < self.batch_size and (event_indices or no_event_indices):
                if event_indices:
                    batch.append(event_indices.pop())
                elif no_event_indices:
                    batch.append(no_event_indices.pop())

            if len(batch) == self.batch_size or (not self.drop_last and batch):
                self.rng.shuffle(batch)
                all_batches.append(batch)

        return iter(all_batches)

    def __len__(self):
        # Max batches limited by the smaller group
        max_pairs = min(len(self.event_indices), len(self.no_event_indices))
        max_batches = (2 * max_pairs) // self.batch_size
        if self.drop_last:
            return max_batches
        else:
            # Might get one extra partial batch
            return max_batches + (1 if (2 * max_pairs) % self.batch_size else 0)
