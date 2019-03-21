import numpy as np
import cv2
import os
import extract_information_flow_field as OF
import time

start = time.time()
folder = 'cyberzoo_ex/'
images = sorted(os.listdir(folder))


# Take first frame

old_frame = cv2.imread(os.path.join(folder,images[0]))

x = 110
w = len(old_frame)-x
y = 40
h = len(old_frame[0]) - y

old_frame_cropped = old_frame[x:w,y:h]


for i in range(1,len(images),1):
    frame = cv2.imread(os.path.join(folder,images[i]))
    # calculate optical flow
    frame_cropped = frame[x:w,y:h]
    points_old, points_new, flow_vectors = OF.determine_optical_flow(old_frame_cropped,frame_cropped) 
    
    # Now update the previous frame and previous points
    old_frame = cv2.imread(os.path.join(folder,images[i]))
    old_frame_cropped = old_frame[x:w,y:h]
    
cv2.destroyAllWindows()

end = time.time()

print((end-start))
print((end-start)/len(images)*1000)