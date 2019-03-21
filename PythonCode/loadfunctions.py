import cv2
import os

def load_images_from_folder(folder,i):
    for filename in sorted(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))
    return im
