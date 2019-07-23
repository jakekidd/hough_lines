from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import cv2, sys

if len(sys.argv) < 2:
	print("Please pass an image filepath as an argument.")
	sys.exit(1)

image_filepath = sys.argv[1]
image = cv2.imread(image_filepath, 0)
edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=10, line_length=1,
                                 line_gap=6)

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[1].set_xlim((0, image.shape[1]))
ax[1].set_ylim((image.shape[0], 0))
ax[1].set_title('Probabilistic Hough Lines')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()