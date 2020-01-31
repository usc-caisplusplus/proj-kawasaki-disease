import numpy as np
import cv2
import os

root = '/content/drive/Shared drives/Kawasaki Disease'

kd_dir = os.path.join(root, 'KD_Images')
non_kd_dir = os.path.join(root, 'Other_Images')

kd_sample_count = 0

kd_dir_subdirs = [res[0] for res in os.walk(kd_dir)]
for subdir in kd_dir_subdirs:
  sample_count = len(os.listdir(subdir))
  print('{} contains {} images'.format(subdir, sample_count))
  kd_sample_count += sample_count

print()
print(kd_sample_count)
