"""NuGraph data module"""
from argparse import ArgumentParser
import warnings

import os
import sys
import h5py
import tqdm
import torch

from torch import tensor, cat
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from itertools import islice, cycle 

from ..data import NuGraphDataset, NuGraphCombinedDataset, NuGraphCombinedDatasetCycle, BalanceSampler

DEFAULT_DATA = ("$NUGRAPH_DATA/uboone-opendata/"
                "uboone-opendata-19be46d89d0f22f5a78641d724c1fedd.gnn.h5")

class NuGraphDataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_source_path: str = "auto",
                 data_target_path: str = None,
                 model: type[torch.nn.Module] = None,
                 batch_size: int = 64,
                 num_workers: int = 5,
                 shuffle: str = 'random',
                 balance_frac: float = 0.1,
                 prepare: bool = False):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        if data_source_path == "auto":
            data_source_path = DEFAULT_DATA
        self.source_filename = os.path.expandvars(data_source_path)
        self.target_filename = None if not data_target_path else os.path.expandvars(data_target_path)
      
        self.batch_size = batch_size
        self.num_workers = num_workers
        if shuffle not in ("random", "balance"):
            print('shuffle argument must be "random" or "balance".')
            sys.exit()
        self.shuffle = shuffle
        self.balance_frac = balance_frac

        filenames = [ self.source_filename, self.target_filename ]
        for file_index, filename in enumerate(filenames):

            if not filename:
               continue
            
            with h5py.File(filename) as f:

                 # load metadata
                 try:
                     # pylint: disable=no-member
                     self.planes = f['planes'].asstr()[()].tolist()
                     self.semantic_classes = f['semantic_classes'].asstr()[()].tolist()
                 except KeyError:
                     print(("Metadata not found in file! "
                            "\"planes\" and \"semantic_classes\" are required."))
                     sys.exit()

                 # get graph structure generation (why is this piece of code missing from the DA code)
                 # if that info is missing, it's first generation
                 try:
                     # pylint: disable=no-member
                     self.gen = f["gen"][()].item()
                 except KeyError:
                     self.gen = 1

                 # load optional event labels
                 if 'event_classes' in f:
                    # pylint: disable=no-member
                    self.event_classes = f['event_classes'].asstr()[()].tolist()
                 else:
                    self.event_classes = None

                 # load sample splits
                 try:
                     # pylint: disable=no-member
                     train_samples = f['samples/train'].asstr()[()]
                     val_samples = f['samples/validation'].asstr()[()]
                     test_samples = f['samples/test'].asstr()[()]
                 except KeyError:
                     print(("Sample splits not found in file! "
                            "Call \"generate_samples\" to create them."))
                     sys.exit()

                 # load data sizes
                 try:
                     self.train_datasize = f['datasize/train'][()]
                 except KeyError:
                     print(("Data size array not found in file! "
                            "Call \"generate_samples\" to create it."))
                     sys.exit()

            transform = model.transform(self.planes) if model else None

            if file_index == 0 :
               self.source_transform = transform
            else:
               self.target_transform = transform
            
            train_dataset = NuGraphDataset(filename, train_samples, self.source_transform )
            val_dataset   = NuGraphDataset(filename, val_samples, self.source_transform)
            test_dataset  = NuGraphDataset(filename, test_samples, self.source_transform)

            if file_index == 0:
               self.train_dataset = train_dataset
               self.val_dataset   = val_dataset
               self.test_dataset  = test_dataset
            elif file_index == 1:
               self.train_dataset_target = train_dataset
               self.val_dataset_target   = val_dataset
               self.test_dataset_target  = test_dataset

        """
        Carry out the procedure for combining the input source and target datasets.
        Note: The NuGraphCombinedCycle functionality has not been fully tested.
               It cycles through the smaller dataset until the larger dataset is loaded.
        """
        if self.target_filename:
           self.combined_train = NuGraphCombinedDataset(self.train_dataset, self.train_dataset_target)
           self.combined_val   = NuGraphCombinedDataset(self.val_dataset, self.train_dataset_target)
           self.combined_test  = NuGraphCombinedDataset(self.test_dataset, self.train_dataset_target)

    @staticmethod
    def generate_samples(data_path: str):
        with h5py.File(data_path) as f:
            samples = list(f['dataset'].keys())
        split = int(0.05 * len(samples))
        splits = [ len(samples)-(2*split), split, split ]
        train, val, test = torch.utils.data.random_split(samples, splits)

        with h5py.File(data_path, "r+") as f:
            for name in [ 'train', 'validation', 'test' ]:
                key = f'samples/{name}'
                if key in f:
                    del f[key]

        with h5py.File(data_path, "r+") as f:
            f.create_dataset("samples/train", data=list(train))
            f.create_dataset("samples/validation", data=list(val))
            f.create_dataset("samples/test", data=list(test))

        with h5py.File(data_path, "r+") as f:
            try:
                planes = f['planes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" is required.')
                sys.exit()

        with h5py.File(data_path, "r+") as f:
            if 'datasize/train' in f:
                del f['datasize/train']
        transform = PositionFeatures(planes)
        dataset = NuGraphDataset(data_path, train, transform)
        def datasize(data):
            ret = 0
            for store in data.stores:
                for val in store.values():
                    ret += val.element_size() * val.nelement()
            return ret
        dsize = [datasize(data) for data in tqdm.tqdm(dataset)]
        del dataset
        with h5py.File(data_path, "r+") as f:
            f.create_dataset('datasize/train', data=dsize)

    @staticmethod
    def collate_func(batch):
        """
            Custom collate function to combine a batch of paired dataset items.
        
            Converts a list of tuples (from NugraphCombinedDataset or 
            NugraphCombinedDatasetCycle) into two batched objects, one 
            for each dataset.
        
            Args:
                batch (list of tuples): Each element is a tuple (dataA, dataB).
        
            Returns:
                tuple: Two batched objects (batchA, batchB) created using 
                       `Batch.from_data_list` for datasetA and datasetB, respectively.
        """
        dataA_list, dataB_list = zip(*batch)  # Unzip dataA and dataB
        batchA = Batch.from_data_list(dataA_list)  # Batch data from datasetA
        batchB = Batch.from_data_list(dataB_list)  # Batch data from datasetB
        return batchA, batchB
    
    def train_dataloader(self) -> DataLoader:      
        if not self.target_filename:
           if self.shuffle == 'balance':
              shuffle = False
              sampler = BalanceSampler.BalanceSampler(
                           datasize=self.train_datasize,
                           batch_size=self.batch_size,
                           balance_frac=self.balance_frac)
           else:
              shuffle = True
              sampler = None
            
           dataloader_train = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                sampler=sampler, drop_last=True,
                                shuffle=shuffle, pin_memory=True)
        else:
            dataloader_train = DataLoader(self.combined_train,
                                  batch_size=self.batch_size, drop_last=True, 
                                  shuffle=True, collate_fn=self.collate_func, pin_memory=True) 
        return dataloader_train
    
    
    def val_dataloader(self) -> DataLoader:
        if not self.target_filename:
           dataloader_val = DataLoader(self.val_dataset, num_workers=self.num_workers,
                                    batch_size=self.batch_size)
        else:    
           dataloader_val = DataLoader(self.combined_val,
                                   batch_size=self.batch_size, collate_fn=self.collate_func,)   
        return dataloader_val    

    
    def test_dataloader(self) -> DataLoader:
        if not self.target_filename:
           dataloader_test = DataLoader(self.test_dataset, num_workers=self.num_workers,
                                 batch_size=self.batch_size)
        else:
           dataloader_test = DataLoader(self.combined_test,
                                  batch_size=self.batch_size, collate_fn=self.collate_func,)
        return dataloader_test
            
    
    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        data = parser.add_argument_group('data', 'Data module configuration')
        data.add_argument('--data-source-path', dest='data-source-path', type=str, default="auto",
                          help='Location of the input source data file')
        data.add_argument('--data-target-path', type=str, default=None,
                          help='Location of the input target data file')
        data.add_argument('--batch-size', type=int, default=64,
                          help='Size of each batch of graphs')
        data.add_argument('--num-workers', type=int, default=5,
                          help='Number of data loader worker processes')
        data.add_argument('--limit_train_batches', type=int, default=None,
                          help='Max number of training batches to be used')
        data.add_argument('--limit_val_batches', type=int, default=None,
                          help='Max number of validation batches to be used')
        data.add_argument('--shuffle', type=str, default='balance',
                          help='Dataset shuffling scheme to use')
        data.add_argument('--balance-frac', type=float, default=0.1,
                          help='Fraction of dataset to use for workload balancing')
        return parser
