import torch
from PIL import Image
from glob import glob
import numpy as np
from torchvision import transforms
from models.jessie.first import main, build, test_transforms
# from models.matt.second import main, build, test_transforms
import cv2
import os

if __name__ == '__main__':

    model_name = 'models/jessie/tongue.pt'
    model, device, criterion, optimizer = build()
    model.load_state_dict(torch.load(model_name))
    model = model.cpu()
    model.eval()

    image = 'testing/positive/tongue/tongue.jpg'
    image = Image.open(image)

    grid_dim = 32

    w, h = image.size
    map = np.zeros((h, w, 1))

    for i in range(0, h - grid_dim, grid_dim):
        print(i)
        x_start = i
        x_end = i + grid_dim

        for j in range(0, w - grid_dim, grid_dim):
            y_start = j
            y_end = j + grid_dim

            cell = image.crop((y_start, x_start, y_end, x_end))
            # discard the transparent, alpha channel (that's the :3) and add the batch dimension
            cell = test_transforms(cell)[:3, :, :].unsqueeze(0)
            result = model(cell)[0, 1]
            result_color = result.cpu().data.numpy()

            map[x_start:x_end, y_start:y_end] = result_color

    map -= map.min()
    print(map.max())
    map *= 255/map.max()
    cv2.imshow('map', map.astype('uint8'))
    cv2.waitKey(0)

