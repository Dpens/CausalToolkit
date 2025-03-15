import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


class Exponential(nn.Module):
    def __init__(self, scale, requires_grad=False):
        super().__init__()
        self._scale = nn.Parameter(
            torch.as_tensor(scale).float().log(), requires_grad=requires_grad
        )

    @property
    def scale(self):
        # return self._scale.abs()
        return self._scale.exp()

    def eval(self, x):
        # return F.hardshrink((-x / self.scale).exp(), 1e-5)
        return (-x / self.scale).clamp(-15, 15).exp()

    def integral(self, x):
        return self.scale - self.scale * self.eval(x)
    
def generate_sequence_mask(lengths, device=None):
    """
    Args:
        lengths (LongTensor): 1-D
    Returns:
        BoolTensor: [description]
    """
    index = torch.arange(lengths.max(), device=device or lengths.device)
    return index.unsqueeze(0) < lengths.unsqueeze(1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self
    
class EventSeqDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
    """

    def __init__(self, event_seqs, min_length=1, sort_by_length=False):

        self.min_length = min_length
        self._event_seqs = [
            torch.FloatTensor(seq)
            for seq in event_seqs
            if len(seq) >= min_length
        ]
        if sort_by_length:
            self._event_seqs = sorted(self._event_seqs, key=lambda x: -len(x))

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        # TODO: can instead compute the elapsed time between events
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(X):
        return nn.utils.rnn.pad_sequence(X, batch_first=True)
    
def convert_to_bucketed_dataloader(
    dataloader: DataLoader, key_fn=None, keys=None, shuffle_same_key=True
):
    """Convert a data loader to bucketed data loader with a given keys.

    Args:
        dataloader (DataLoader):
        key_fn (Callable]):  function to extract keys used for constructing
          the bucketed data loader; should be of the same key as the
          dataset. Only
        keys (List): keys used for sorting the elements in the dataset.
        shuffle_same_key (bool, optional): Whether to shuffle the instances of
          the same keys. Defaults to False.

    Returns:
        DataLoader:
    """

    assert (
        dataloader.batch_size is not None
    ), "The `batch_size` must be present for the input dataloader"

    dataset = dataloader.dataset
    assert (key_fn is None) != (
        keys is None
    ), "Only either `key_fn` or `keys` can be set."

    if key_fn is not None:
        keys = [key_fn(dataset[i]) for i in range(len(dataset))]
    else:
        assert len(keys) == len(dataset)

    batch_sampler = KeyBucketedBatchSampler(
        keys,
        batch_size=dataloader.batch_size,
        drop_last=dataloader.drop_last,
        shuffle_same_key=shuffle_same_key,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
    )


class KeyBucketedBatchSampler(torch.utils.data.Sampler):
    """Pseduo bucketed batch sampler.

    Sample in a way that
    Args:
        keys (List[int]): keys by which the same or nearby keys are allocated
          in the same or nearby batches.
        batch_size (int):
        drop_last (bool, optional): Whether to drop the last incomplete batch.
          Defaults to False.
        shuffle_same_key (bool, optional): Whether to shuffle the instances of
          the same keys. Defaults to False.
    """

    def __init__(
        self, keys, batch_size, drop_last=False, shuffle_same_key=True
    ):

        self.keys = keys
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_same_key = shuffle_same_key

        # bucket sort; maintain random order inside each bucket
        buckets = {}
        for i, key in enumerate(self.keys):
            if key not in buckets:
                buckets[key] = [i]
            else:
                buckets[key].append(i)

        self.buckets = buckets

    def __iter__(self):
        indices = []
        for key in sorted(self.buckets.keys()):
            v = self.buckets[key]
            if self.shuffle_same_key:
                random.shuffle(v)
            indices += v

        index_batches = []
        for i in range(0, len(indices), self.batch_size):
            j = min(i + self.batch_size, len(indices))
            index_batches.append(indices[i:j])
        del indices

        if self.drop_last and len(index_batches[-1]) < self.batch_size:
            index_batches = index_batches[:-1]

        random.shuffle(index_batches)
        for indices in index_batches:
            yield indices

    def __len__(self):
        if self.drop_last:
            return len(self.keys) // self.batch_size
        else:
            return (len(self.keys) + self.batch_size - 1) // self.batch_size
        

def split_dataloader(dataloader, ratio: float):
    dataset = dataloader.dataset
    n = len(dataset)
    lengths = [int(n * ratio), n - int(n * ratio)]
    datasets = torch.utils.data.random_split(dataset, lengths)

    copied_fields = ["batch_size", "num_workers", "collate_fn", "drop_last"]
    dataloaders = []
    for d in datasets:
        dataloaders.append(
            DataLoader(
                dataset=d, **{k: getattr(dataloader, k) for k in copied_fields}
            )
        )

    return tuple(dataloaders)


def compare_metric_value(val1: float, val2: float, metric: str) -> bool:
    """Compare whether val1 is "better" than val2.

    Args:
        val1 (float):
        val2 (float): can be NaN.
        metric (str): metric name

    Returns:
        (bool): True only if val1 is better than val2.
    """
    from math import isnan

    if isnan(val2):
        return True
    elif isnan(val1):
        return False
    elif metric == "acc":
        return val1 > val2
    elif metric == "nll":
        return val1 < val2
    else:
        raise ValueError(f"Unknown metric={metric}.")
