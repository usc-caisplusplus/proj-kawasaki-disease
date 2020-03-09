import numpy as np
import torch
import torch.nn as nn
from models.util import create_methods
from torchvision import transforms, models

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

    return device, model, criterion, optimizer


def train_eval_loop(image_dir, out_path, epochs=10):
    train, eval, save = create_methods(image_dir, train_transforms, build)

    min_validation_loss = np.Inf
    for epoch in range(epochs):
        loss_train = train()
        loss_eval, accuracy = eval()

        if loss_eval <= min_validation_loss:
            save(out_path)
            min_validation_loss = loss_eval

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            epoch + 1, loss_train, loss_eval, accuracy))
