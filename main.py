import torch
from PIL import Image
from glob import glob
import numpy as np
from models.jessie.first import main, build, test_transforms
# from models.matt.first import main, build, test_transforms

dir = 'dataset/eyes'
model_name = 'models/jessie/eye.pt'

"""TRAINING"""
main(dir, model_name)

"""TESTING"""
def load_image(path):
    image = Image.open(path)
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = test_transforms(image)[:3, :, :].unsqueeze(0)
    return image

def predict(path, model):
    image = load_image(path)
    model = model.cpu()
    model.eval()
    return torch.argmax(model(image))

model, device, criterion, optimizer = build()
model.load_state_dict(torch.load(model_name))

pos = np.array(glob(dir + '/yes/*'))
neg = np.array(glob(dir + '/no/*'))

true_pos = 0
false_neg = 0

for i in range(pos.shape[0]):
    path = pos[i]
    pred = predict(path, model)
    if pred == 1:
        true_pos += 1
    else:
        false_neg += 1

print('True Positives: {}\tFalse Negatives: {}'.format(true_pos, false_neg))

true_neg = 0
false_pos = 0

for i in range(neg.shape[0]):
    path = neg[i]
    pred = predict(path, model)
    if pred == 1:
        false_pos += 1
    else:
        true_neg += 1

print('False Positives: {}\tTrue Negatives: {}'.format(false_pos, true_neg))


# TODO!!
# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
def make_weights_for_balanced_classes(images, nclasses):
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