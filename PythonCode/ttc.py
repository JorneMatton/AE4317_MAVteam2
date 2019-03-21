import cv2
import os
import numpy as np

folder = 'cyberzoo_ex/'
images = sorted(os.listdir(folder))
img = cv2.imread(os.path.join(folder,images[0]))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20,20),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))


# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

fps = 30
timefactor = int(fps/15)

# find and draw the keypoints
for i in range(1,len(images),timefactor):
    
    kp = fast.detect(img,None)
    
    img2 = cv2.imread(os.path.join(folder,images[i]))
    img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))
    

#    points_new,status,error_match = cv2.calcOpticalFlowPyrLK(img, img2, points_old, points_old, **lk_params)
#    
#    list1 = []
#    list2 = []
#    
#    for j in range(len(status)):
#        if status[j] == 1:
#            list1.append(points_old[j])
#            list2.append(points_new[j])
#            
#    list1 = np.asarray(list1)
#    list2 = np.asarray(list2)
   
    
#    flow_vectors = list2 - list1;
#   #################################################################################### 
#    im = (0.5 * img.copy().astype(float) + 0.5 * img2.copy().astype(float)) / 255.0;
#    n_points = len(points_old);
#    color = (0.0,1.0,0.0);
#    for p in range(n_points):
#        cv2.arrowedLine(im, tuple(points_old[p]), tuple(points_new[p]), color)
    ######################################################################################3
    
    img = cv2.imread(os.path.join(folder,images[i]))
    
    cv2.imshow('Flow', img2);
    cv2.waitKey(int((1000/fps)*timefactor));
cv2.destroyAllWindows()