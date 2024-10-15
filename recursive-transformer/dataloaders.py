import tiktoken
import torch
from torch.utils.data import Sampler, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import tiktoken


class XorDataset(Dataset):
    def __init__(self, n):
        binary_list = [0, 1, 1] * (n // 3)
        self.tokens = torch.tensor(binary_list)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

class TextDataset(Dataset):
    def __init__(self, file_path, encoding='gpt2'):
        with open(file_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding(encoding)
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

class ParallelBatchSampler(Sampler):
    def __init__(self, data_source, T, B):
        # B workers move forward in parallel and extract segments of length
        self.data_source = data_source
        self.T = T
        self.B = B # number of workers
        self.total_length = len(data_source)
        self.worker_job_length = self.total_length // B # in characters
        self.n_sequences_per_worker = self.worker_job_length // self.T # in sequences

    def __iter__(self):
        for sequence_nb in range(self.n_sequences_per_worker):
            x_indices = []
            y_indices = []
            print(sequence_nb)
            for worker_nb in range(self.B):
                start = worker_nb * self.worker_job_length + (sequence_nb * self.T)
                end = start + self.T
                if end <= self.total_length:
                    x_indices.extend(range(start, end))
                    y_indices.extend(range(start+1, end+1))
            if x_indices and y_indices:
                yield torch.tensor(x_indices).view(self.B, self.T), torch.tensor(y_indices).view(self.B, self.T)

    def __len__(self):
        return self.n_sequences_per_worker * self.T


class SequentialBatchSampler(Sampler):
    def __init__(self, data_source, T, B):
        self.data_source = data_source
        self.batch_size = T * B
        self.B = B
        self.T = T

    def __iter__(self):
        start = 0
        while start < len(self.data_source):
            end = min(start + self.batch_size + 1, len(self.data_source))
            indices = torch.arange(start, end)
            x = indices[:-1].view(self.B, self.T)
            y = indices[1:].view(self.B, self.T)
            yield x, y
            start += self.batch_size

    def __len__(self):
        return (len(self.data_source) - 1) // self.batch_size


class DatasetLoader:
    def __init__(self, B, T, dataset, sampler='sequential'):
        self.B = B  # Number of parallel batches
        self.T = T  # Sequence length
        print(f"Initializing DatasetLoader with B={B}, T={T}")

        self.dataset = dataset
        if sampler == 'sequential':
            self.sampler = SequentialBatchSampler(self.dataset, T, B)
        elif sampler == 'parallel':
            self.sampler = ParallelBatchSampler(self.dataset, T, B)

        self.dataloader = DataLoader(self.dataset, batch_sampler=self.sampler)

        print(f"Loaded {len(self.dataset)} tokens")
        print(f"1 epoch = {len(self.sampler)} batches")
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        x, y = next(self.iterator)
        return x, y
