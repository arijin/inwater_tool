import numpy as np
import cv2
img = np.zeros((1000, 1000, 3), dtype=np.uint8)
points = np.random.randint(0, 1000, (10000, 2))
rows = points[:, 1]
cols = points[:, 0]
print(rows.shape)
img[rows, cols] = (255, 0, 255)

cv2.imshow("point cloud", img)
cv2.waitKey(0)
