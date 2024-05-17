import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, Resize, ToTensor, Compose
from pdo import pdo_net
from PIL import Image
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='net')
parser.add_argument('--model', '-a', default='R', help='Regular(R) or Quotient(Q)')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate (default: 2e-3)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--order', default=4, type=int, help='differential operator order')
parser.add_argument('--bias', default=False, type=bool, help='bias')
parser.add_argument('--reduction', default=1, type=float, help='reduction_ratio')
parser.add_argument('--g', default=4, type=int, help='g * q = z, q: partition number z: number of input fields')
parser.add_argument('--s', default=4, type=int, help='slice, only 1 is valid for quotient model')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--dis', default='gauss', help='discretization: fd, gauss')
parser.add_argument('--flip', default=False, type=bool, help="D16|5C16 or C16")

args = parser.parse_args()
device = ('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

# Set random seeds
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

print(args)


def compute_param(net):
    """Compute the number of parameters in the network"""
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class MnistRotDataset(Dataset):
    """MNIST Rotation Dataset"""

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
        file = "data/mnist_all_rotation_normalized_float_train_valid.amat" if mode == "train" else "data/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform
        data = np.loadtxt(file, delimiter=' ')
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


# Transforms
train_transform = Compose([
    Resize(84),
    RandomRotation(180),
    Resize(28),
    ToTensor()
])

test_transform = Compose([
    ToTensor(),
])

# Datasets and Dataloaders
mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size)

# Model selection
quotient_type = (('regular', 1), ('quo_4', 1))
channel_q = [12, 16, 24, 24, 32, 64]
channel_r = [12, 16, 24, 24, 32, 64] if args.flip else [16, 24, 32, 32, 48, 64]

if args.model == "R":
    net = pdo_net(16, flip=args.flip, dis=args.dis, g=args.g, reduction_ratio=args.reduction, drop=args.dropout,
                  s=args.s, order=args.order, channel=channel_r, type='regular')
elif args.model == "Q":
    net = pdo_net(16, False, dis=args.dis, g=args.g, reduction_ratio=args.reduction, drop=args.dropout, s=args.s,
                  order=args.order, type=quotient_type, channel=channel_q)

param = compute_param(net)
model = nn.DataParallel(net).to(device)
loss_function = nn.CrossEntropyLoss().to(device)

# Learning rate scheduler
schedule = [60, 120, 150, 180]
learning_rate = args.learning_rate
best = 0.0

for epoch in range(1, args.epochs + 1):
    if epoch in schedule:
        learning_rate *= 0.1 if epoch == schedule[0] else 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    model.train()
    total = 0
    correct = 0
    print(f"Parameters of the net: {param / (10 ** 3):.2f}K")

    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x, t = x.to(device), t.to(device)
        y = model(x)
        _, prediction = torch.max(y.data, 1)
        total += t.size(0)
        correct += (prediction == t).sum().item()
        loss = loss_function(y, t)
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch} | train accuracy: {correct / total * 100:.2f}%")

    # Evaluate on test set
    if epoch >= schedule[0]:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for x, t in test_loader:
                x, t = x.to(device), t.to(device)
                y = model(x)
                _, prediction = torch.max(y.data, 1)
                total += t.size(0)
                correct += (prediction == t).sum().item()

        test_accuracy = correct / total * 100
        print(f"epoch {epoch} | test accuracy: {test_accuracy:.2f}%")
        if test_accuracy > best:
            best = test_accuracy

    print('\n')

print(f'Best test acc: {best:.2f}%')
