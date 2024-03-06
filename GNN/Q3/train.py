from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np

from models import *
from sample import Sampler


# random seed setting
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def accuracy(pred_labels, real_labels):
    accuracy = (pred_labels.argmax(axis=1)==real_labels).sum()/len(pred_labels)
    return accuracy.item()

# define the training function.
def train_one_epoch(model, optimizer, labels, adj, features, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train


def evaluate(model, idx_test, labels, adj, features):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test


def train(model, sampler, optimizer, scheduler, epochs, normalization, sampling_percent):
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes()
    all_train_acc, all_val_acc = [], []
    for epoch in (pbar := tqdm(range(1, epochs+1))):
        train_adj, train_fea = sampler.randomedge_sampler(sampling_percent, normalization)
        loss_train, train_acc = train_one_epoch(model, optimizer, labels, train_adj, train_fea, idx_train)
        
        val_adj, val_fea = sampler.get_val_set(normalization)
        val_acc = evaluate(model, idx_val, labels, val_adj, val_fea)

        if scheduler:
            scheduler.step()
            all_train_acc.append(train_acc)
            all_val_acc.append(val_acc)
        
        metrics = "Epoch {} | train_loss: {:.3f} | train_acc: {:.3f} | val_acc: {:.3f}".format(epoch, loss_train, train_acc, val_acc)
        pbar.set_description(f"{metrics}")
    test_adj, test_fea = sampler.get_test_set(normalization)
    test_acc = evaluate(model, idx_test, labels, test_adj, test_fea)
    return all_train_acc, all_val_acc, test_acc

def plot_accuracy(all_train_acc, all_val_acc, title, save_path, figure_size=(20,3)):
    plt.rcParams["figure.figsize"] = figure_size
    plt.plot(all_train_acc, label="train")
    plt.plot(all_val_acc, label="val")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    sampler = Sampler(args.dataset, args.device)
    # The model
    if args.new_model:
        model = NewGCNModel(sampler.nfeat, sampler.nclass, args.hidden, args.num_layers, args.dropout, args.batch_norm, args.skip_connection)
    else:
        model = GCNModel(sampler.nfeat, sampler.nclass, args.hidden, args.num_layers, args.dropout, args.batch_norm, args.skip_connection)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

    all_train_acc, all_val_acc, test_acc = train(model, sampler, optimizer, scheduler, args.epochs, args.normalization, args.sampling_percent)
    plot_accuracy(all_train_acc, all_val_acc, "DropEdge", f"{args.result_path}", figure_size=(10, 8))
    print(f"accuracy on test dataset: {test_acc}")
    


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    # Training parameter 
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epochs', type=int, default=800,  help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true', default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dataset', default="CuraFull", help="The data set: CuraFull or CiteSeer")
    parser.add_argument('--result_path', default="results/accuracy.png", help="Path to save results plots")

    # Model parameter
    parser.add_argument('--new_model', action='store_true', default=False, help='Use new idea in model')
    parser.add_argument('--skip_connection', action='store_true', default=False, help='Use skip connection in model or not')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_norm', action='store_true', default=False, help='Enable Bath Norm GCN')
    # parser.add_argument('--withloop', action="store_true", default=False, help="Enable loop layer GCN")
    parser.add_argument("--normalization", default="AugNormAdj", help="The normalization on the adj matrix.")
    parser.add_argument("--num_layers", type=int, default=2, help="The number of layers")
    parser.add_argument("--sampling_percent", type=float, default=1.0, help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    args = parser.parse_args()

    main(args)
