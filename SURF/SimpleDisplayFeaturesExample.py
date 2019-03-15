import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

images = [cv2.imread(file) for file in sorted(glob.glob('*.jpg'))]
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    (kps,descs) = surf.detectAndCompute(image,None)
    img2 = cv2.drawKeypoints(image,kps,None,(255,0,0))
    (h,w) = img2.shape[:2]
    center = (w / 2, h /2)
    M = cv2.getRotationMatrix2D(center, 90, 1)
    img2 = cv2.warpAffine(img2,M,(w,h))
    plt.imshow(img2)
    plt.pause(.001)
    plt.draw()


