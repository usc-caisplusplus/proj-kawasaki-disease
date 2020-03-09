import numpy as np
import torch
import torch.nn as nn
from models.util import create_methods
from torchvision import transforms, models

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = train_transforms


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
