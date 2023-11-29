import argparse
import os

import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AudioClassifier
from dataloader import AudioDataset

def seed_it(seed):
    """
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_one_epoch(model, optimizer, train_loader, epoch, total_epoch, criterion, device):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param epoch:
    :param total_epoch:
    :param criterion:
    :param device:
    :return:
    """
    for batch_idx, (batch_input, label) in enumerate(tqdm(train_loader)):
        batch_input, label = batch_input.to(device), label.to(device)

        prediction = model(batch_input)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{total_epoch}], Step [{batch_idx}/{len(train_loader)}, Loss: {loss.item()}')


def eval_one_epoch(model, epoch, dataloader, criterion, device):
    """
    :param model:
    :param epoch:
    :param dataloader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_input, label in dataloader:
            batch_input, label = batch_input.to(device), label.to(device)
            prediction = model(batch_input)
            loss = criterion(prediction, label)
            total_loss += loss.item()

            _, predicted = torch.max(prediction.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{args.epochs}] - Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}')
    return avg_loss, accuracy


def main(args):

    seed_it(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = AudioDataset(csv_file='data/whale-detection-challenge/whale_data/data/custom_train.csv',
                                 data_dir='data/whale-detection-challenge/whale_data/data/train/', model_name=args.model_name, debug=args.debug)
    val_dataset = AudioDataset(csv_file='data/whale-detection-challenge/whale_data/data/custom_val.csv', data_dir='data/whale-detection-challenge/whale_data/data/train/',
                               model_name=args.model_name, debug=args.debug)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = AudioClassifier(args.model_name, args.dropout).to(device)
    print(model)

    for n, p in model.named_parameters():
        if ("backbone" in n):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable params", n_parameters)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train and evaluate model
    for epoch in tqdm(range(args.epochs)):
        train_one_epoch(model, optimizer, dataloader_train, epoch, args.epochs, criterion, device)
        print(f"Epoch {epoch} done")
        torch.save(model.state_dict(), os.path.join('output', 'model.pth'))
        eval_one_epoch(model, epoch, dataloader_val, criterion, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="batch size used for training")
    parser.add_argument("--epochs", default=10, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout")
    parser.add_argument("--seed", default=1337, type=int, help="seed")
    parser.add_argument("--model_name", default="laion/clap-htsat-fused", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
