import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(module), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    module = [
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks):
        module.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    module.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*module)


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()
    losses = []
    correct = 0
    for x, y in dataloader:
        y_pred = model(x)
        loss = nn.SoftmaxLoss()(y_pred, y)
        losses.append(loss.numpy())
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        correct += (y_pred.numpy().argmax(axis=1) == y.numpy()).sum()
    return 1 - correct / len(dataloader.dataset), np.mean(losses)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )
    data_loader = ndl.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )
    test_data_loader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_blocks=3, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err, train_loss = 0, 0
    test_err, test_loss = 0, 0
    for i in range(epochs):
        start = time.time()
        train_err, train_loss = epoch(data_loader, model, opt)
        test_err, test_loss = epoch(test_data_loader, model)
        end = time.time()
        print("Epoch %d: Train err: %f, Train loss: %f | Test err: %f, Test loss: %f, Time: %f" % (
            i, train_err, train_loss, test_err, test_loss, end - start
        ))
    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
