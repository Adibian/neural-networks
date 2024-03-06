import torch
from torch.optim import Adam
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def evaluate_model(model, data, indexes):
    model.eval()
    with torch.no_grad():
        predicted = model(data)
        pred_labels, real_labels = predicted[indexes], data["labels"][indexes]
        accuracy = (pred_labels.argmax(axis=1)==real_labels).sum()/len(pred_labels)
    return accuracy.item()

def train_model(model, dataset, device, epoches=100, lr=0.005, weight_decay=0.001):
    # define neural network
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    # perform training
    train_indexes, val_indexes = dataset["train_indexes"], dataset["val_indexes"]
    data = {"nodes":dataset["nodes"].to(device), "edges":dataset["edges"].to(device), "labels":dataset["labels"].to(device)}

    train_accuracy, val_accuracy = [], []
    for epoch in (pbar := tqdm(range(1, epoches+1))):
        ## train model
        model.train()
        optimizer.zero_grad()

        train_predicted = model(data)
        train_pred_labels, train_real_labels = train_predicted[train_indexes], data["labels"][train_indexes]
        loss = loss_function(train_pred_labels, train_real_labels)
        loss.backward()
        optimizer.step()
        train_acc = (train_pred_labels.argmax(axis=1)==train_real_labels).sum()/len(train_pred_labels)
        train_accuracy.append(train_acc.item())

        ## validate model
        val_acc = evaluate_model(model, data, val_indexes)
        val_accuracy.append(val_acc)
        metrics = "Epoch {} | train_loss: {:.3f} | train_acc: {:.3f} | val_acc: {:.3f}".format(epoch, loss.item(), train_acc, val_acc)
        pbar.set_description(f"{metrics}")

    return model, train_accuracy, val_accuracy

def plot_accuracy(plot_data, num_rows, num_cols, figure_size=(20,3)):
    plt.rcParams["figure.figsize"] = figure_size
    for plot_num in range(num_rows*num_cols):
        data = plot_data[plot_num]
        for row in data["data"]:
            plt.subplot(num_rows, num_cols, plot_num+1)
            plt.plot(row["accuracy"], label=row["name"])
        plt.legend()
        plt.title(data["title"])
        plt.xlabel("Epoches")
        plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    plt.close()