"""NuGraph dataset"""
from typing import Callable, Optional
import h5py
from torch_geometric.data import Dataset
from pynuml.data import NuGraphData

class NuGraphDataset(Dataset):
    """NuGraph dataset

    Args:
        filename: Name of dataset file
        samples: List of graph object dataset names in file
        transform: Transforms to apply to graph objects
    """
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)

        self.filename = filename
        self.samples = samples

        # the HDF5 file is opened on first use in each process (see `file`)
        self._file = None
        self._file_pid = None

    @property
    def file(self) -> h5py.File:
        """
        Read-only HDF5 handle, opened lazily and once per process.

        A handle opened in the parent and inherited by forked DataLoader
        workers is shared across processes, which h5py does not support;
        reopening in each worker makes num_workers > 0 safe.
        """
        if self._file is None or self._file_pid != os.getpid():
           self._file = h5py.File(self.filename, "r")
           self._file_pid = os.getpid()
        return self._file

    def __getstate__(self):
        # workers started with "spawn" receive a pickled copy: send no handle
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state    

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        key = f"/dataset/{self.samples[idx]}"
        return NuGraphData.load(self.file[key])
        
class NuGraphCombinedDataset(Dataset):
    """
    A dataset that pairs two datasets together and returns corresponding items.
    Iteration stops at the length of the shorter dataset.
    """
    
    def __init__(self, datasetA, datasetB):
        """
        Initialize the combined dataset.

        Args:
            datasetA (Dataset): First dataset.
            datasetB (Dataset): Second dataset.
        """
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        """
        Return the length of the combined dataset.

        Returns:
            int: Minimum length of the two datasets.
        """
        return min(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        """
        Retrieve a paired item from both datasets at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing (item_from_datasetA, item_from_datasetB)
        """
        dataA = self.datasetA[idx]
        dataB = self.datasetB[idx]
        return dataA, dataB 


class NuGraphCombinedDatasetCycle(Dataset):
    """
    A dataset that pairs two datasets together and cycles over the shorter dataset.
    Iteration continues for the length of the longer dataset by wrapping around the shorter one.
    """
    def __init__(self, datasetA, datasetB):
        """
        Initialize the combined cyclic dataset.

        Args:
            datasetA (Dataset): First dataset.
            datasetB (Dataset): Second dataset.
        """
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        """
        Return the length of the combined cyclic dataset.

        Returns:
            int: Maximum length of the two datasets.
        """
        return max(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        """
        Retrieve a paired item from both datasets at the given index,
        cycling over the shorter dataset if necessary.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing (item_from_datasetA, item_from_datasetB)
        """
        dataA = self.datasetA[idx % len(self.datasetA)]
        dataB = self.datasetB[idx % len(self.datasetB)]
        return dataA, dataB
