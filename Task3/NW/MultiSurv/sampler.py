import torch
from torch.utils.data import Sampler
import numpy as np
import random

class EventBalancedBatchSampler(Sampler):
    def __init__(self, events, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.events = np.array(events)
        self.event_indices = np.where(self.events == 1)[0].tolist()
        self.no_event_indices = np.where(self.events == 0)[0].tolist()

        if len(self.event_indices) == 0:
            raise ValueError("No event samples found to create balanced batches.")

        self.num_batches = len(self.events) // batch_size

    def __iter__(self):
        event_indices = self.event_indices.copy()
        no_event_indices = self.no_event_indices.copy()

        random.shuffle(event_indices)
        random.shuffle(no_event_indices)

        all_indices = []

        while len(event_indices) > 0 and len(no_event_indices) + len(event_indices) >= self.batch_size:
            batch = []

            # Ensure at least one event in each batch
            if len(event_indices) > 0:
                batch.append(event_indices.pop())

            # Fill the rest with no-event samples
            while len(batch) < self.batch_size and no_event_indices:
                batch.append(no_event_indices.pop())

            # If still space, fill with events if available
            while len(batch) < self.batch_size and event_indices:
                batch.append(event_indices.pop())

            if len(batch) == self.batch_size or not self.drop_last:
                random.shuffle(batch)
                all_indices.append(batch)

        return iter(all_indices)

    def __len__(self):
        return self.num_batches if self.drop_last else (len(self.events) + self.batch_size - 1) // self.batch_size
