from torch_geometric.datasets import CoraFull, Planetoid
import torch


seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def get_datasets():
    """ Get datasets CoraFull and CiteSeer from torch_geometric """
    CoraFull_dataset = CoraFull(root='data')
    CiteSeer_dataset = Planetoid(root='data', name='CiteSeer')
    new_datasets = []
    for dataset in [CoraFull_dataset, CiteSeer_dataset]:
        new_dataset = {"name": dataset.name, "num_classes": dataset.num_classes, "nodes":dataset.x, "edges": dataset.edge_index.T, "labels": dataset.y}
        new_datasets.append(new_dataset)
    return new_datasets[0], new_datasets[1]

def show_data_info(datasets):
    """ Print some information from input datasets """
    for dataset in datasets:
        print(f"dataset name: {dataset['name']}")
        print(f"Number of nodes: {dataset['nodes'].shape[0]}")
        print(f"Number of edges: {dataset['edges'].shape[0]}")
        print(f"Number of classes: {dataset['num_classes']}")
        print(f"Number of features: {dataset['nodes'].shape[1]}\n")

def split_data(datasets):
    """ Split of any classes to train, val, test parts """
    new_datasets = []
    for dataset in datasets:
        classes_indexes = {i:[] for i in list(set(dataset["labels"].tolist()))}
        for index, label in enumerate(dataset["labels"]):
            classes_indexes[label.item()].append(index)
        train_indexes, val_indexes, test_indexes = [], [], []
        for indexes in classes_indexes.values():
            indexes = torch.tensor(indexes)[torch.randperm(len(indexes))]
            train_indexes.append(indexes[:int(len(indexes)*0.7)])
            val_indexes.append(indexes[int(len(indexes)*0.7):int(len(indexes)*0.8)])
            test_indexes.append(indexes[int(len(indexes)*0.8):])

        train_indexes, val_indexes, test_indexes = torch.cat(train_indexes), torch.cat(val_indexes), torch.cat(test_indexes)
        dataset["train_indexes"], dataset["val_indexes"], dataset["test_indexes"] = train_indexes, val_indexes, test_indexes
        new_datasets.append(dataset)
    return new_datasets[0], new_datasets[1]
