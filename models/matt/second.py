import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = train_transforms
validation_transforms = train_transforms

def make_weights_for_balanced_classes(dataset, indices):
    nclasses = len(dataset.classes)
    images = [dataset[i] for i in indices]
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def build():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)

    return model, device, criterion, optimizer

def main(image_dir, out_path):
    dataset = datasets.ImageFolder(image_dir, transform=train_transforms)

    valid_pct = 0.2

    train_size = int(np.floor(len(dataset) * (1-valid_pct)))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    weights = make_weights_for_balanced_classes(dataset, train_dataset.indices)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    loader_train = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=0)
    loader_valid = DataLoader(valid_dataset, batch_size=4, num_workers=0, shuffle=True)

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
