import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
        'Cora': (datasets.Planetoid, 'Cora'),
        'Citeseer': (datasets.Planetoid, 'Citeseer'),
        'Pubmed': (datasets.Planetoid, 'Pubmed'),
        'ogbn-arxiv': (PygNodePropPredDataset, 'ogbn-arxiv')
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    if name[:4]=="ogbn":
        print("Downloading " + str(name))
        dataset = dataset_class(root=root, name=name)
    else:
        dataset = dataset_class(root, name=name, transform=transform)

    return dataset


def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = datasets.WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)


class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)
