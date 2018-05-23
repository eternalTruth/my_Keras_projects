"""
输出一个彩色图像的直方图
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt  # img = cv2.imread('/home/aobo/Pictures/IMG_4137.JPG')
img = cv2.imread('jinmao.JPG')
color = ('b', 'g', 'r')
# enumerate():python里的一个新函数
# 它的作用：同时遍历索引(i)和元素(col)
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
