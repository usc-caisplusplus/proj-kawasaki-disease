import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def load_dataset(image_dir, transform, size_test=0.1, size_valid=0.2):
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    size_train = len(dataset)

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

    return loader_train, loader_test, loader_valid

def create_methods(image_dir, transforms, build_method):
    loader_train, loader_test, loader_valid = load_dataset(image_dir, transforms)
    device, model, criterion, optimizer = build_method()

    def train():
        model.train()
        total_loss = 0.0
        for inputs, labels in loader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader_train)

    def eval():
        model.eval()
        total_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in loader_valid:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return total_loss / len(loader_valid), accuracy / len(loader_valid)

    def save(out_path):
        torch.save(model.state_dict(), out_path)

    return train, eval, save
