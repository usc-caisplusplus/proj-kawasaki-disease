# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:10:54 2020

@author: shuya
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),  # ImageNet standards
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    fc = nn.Sequential(
        nn.Linear(1024, 460),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(460, 2),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = fc

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)

    return model, device, criterion, optimizer


def main(image_dir, out_path):
    dataset = datasets.ImageFolder(image_dir, transform=train_transforms)
    size_train = len(dataset)
    size_test = 0.1
    size_valid = 0.2

    split_test = int(np.floor((size_test + size_valid) * size_train))
    split_valid = int(np.floor(size_valid * size_train))

    indices = list(range(size_train))
    np.random.shuffle(indices)

    idx_train = indices[split_test:]
    idx_test = indices[split_valid:split_test]
    idx_valid = indices[:split_valid]

    sampler_train = SubsetRandomSampler(idx_train)
    sampler_test = SubsetRandomSampler(idx_test)
    sampler_valid = SubsetRandomSampler(idx_valid)

    loader_train = DataLoader(dataset, batch_size=32, sampler=sampler_train, num_workers=0)
    loader_test = DataLoader(dataset, batch_size=32, sampler=sampler_test, num_workers=0)
    loader_valid = DataLoader(dataset, batch_size=32, sampler=sampler_valid, num_workers=0)

    model, device, criterion, optimizer = build()

    """TRAINING------------------------------------------------------------------"""
    epochs = 10
    min_validation_loss = np.Inf
    for epoch in range(epochs):
        model.train()

        loss_train = 0.0
        loss_valid = 0.0
        for inputs, labels in loader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        model.eval()

        with torch.no_grad():
            accuracy = 0
            for inputs, labels in loader_valid:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_valid += loss.item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        loss_train = loss_train / len(loader_train)
        loss_valid = loss_valid / len(loader_valid)
        accuracy = accuracy / len(loader_valid)

        if loss_valid <= min_validation_loss:
            torch.save(model.state_dict(), out_path)
            min_validation_loss = loss_valid

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            epoch + 1, loss_train, loss_valid, accuracy))

    """TESTING------------------------------------------------------------------"""