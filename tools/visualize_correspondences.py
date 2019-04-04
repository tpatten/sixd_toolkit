from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer
from params.dataset_params import get_dataset_params

rgb1 = inout.load_im('image1.jpg')
rgb2 = inout.load_im('image2.jpg')
coords1 = np.load('coords1.npy')
coords2 = np.load('coords2.npy')

unravelled1 = coords1.reshape(coords1.shape[0]*coords1.shape[1], coords1.shape[2])/1000
unravelled2 = coords2.reshape(coords2.shape[0]*coords2.shape[1], coords2.shape[2])/1000
indices = unravelled1.copy()
count = 0
for i in range(0, coords1.shape[0]):
    for j in range(0, coords1.shape[1]):
        indices[count] = (i,j,0)
        count += 1
mask1 = np.where(unravelled1.sum(axis=1) > 0)
mask2 = np.where(unravelled2.sum(axis=1) > 0)

print 'coords1.shape', coords1.shape
print 'unravelled1.shape', unravelled1.shape

eps = 0.0002
correspondences = np.zeros((len(mask1[0]),4))
count = 0
for i in mask1[0]:
    for j in mask2[0]:
        dist = np.linalg.norm(unravelled1[i] - unravelled2[j])
        if dist < eps:
            coords1_ix = (int(indices[i][1]), int(indices[i][0]))
            coords2_ix = (int(indices[j][1]), int(indices[j][0]))
            #print 'Found correspondence ', coords1_ix, coords2_ix
            correspondences[count] = (int(indices[i][1]), int(indices[i][0]),\
                                      int(indices[j][1]), int(indices[j][0]))
            count += 1

corr_img = np.hstack((rgb1, rgb2))
width = coords1.shape[1]
height = coords1.shape[0]
for i in range(count):
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    pt1 = (int(correspondences[i][0]), int(correspondences[i][1]))
    pt2 = (int(correspondences[i][2] + width), int(correspondences[i][3]))
    cv2.circle(corr_img, pt1, 1, (b, r, g), 1)
    cv2.circle(corr_img, pt2, 1, (b, r, g), 1)
    cv2.line(corr_img, pt1, pt2, (b, r, g), 1)

vis_img = np.vstack((np.hstack((rgb1, rgb2)), corr_img))
plt.imshow(vis_img),plt.show()
