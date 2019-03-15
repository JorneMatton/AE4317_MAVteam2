import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


DIST_TRESHOLD = 0.1 #euclidean distance threshold for the descriptor match filter
N_SKIP = 1 # gap between preceding matching images

images = [cv2.imread(file) for file in sorted(glob.glob('*.jpg'))]

for idx,newImage in enumerate(images):

    #obtain Surf features
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    (newKps,newDescs) = surf.detectAndCompute(newImage,None)    

    if idx > 0:

        #Match with previous newImage
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)  
        matches = bf.match(prevDescs,newDescs) #  matches = bf.radiusMatch(prevDescs,newDescs, DIST_TRESHOLD)

        #Filter out "worse" matches
        matches = [match for match in matches if match.distance < DIST_TRESHOLD]
        print(len(matches))

        #Filter out matches whose size has decreased or stayed the same
        finalizedMatches =[]
        finalizedKeypoints =[]
        for idx in range(len(matches)):            
            prevKeyPointIdx = matches[idx].queryIdx
            newkeyPointIdx = matches[idx].trainIdx
            sizePrevKeyPoint = prevKps[prevKeyPointIdx].size
            sizeNewKeyPoint = newKps[newkeyPointIdx].size

            if sizeNewKeyPoint > sizePrevKeyPoint:
                finalizedMatches.append(matches[idx])            
                finalizedKeypoints.append(newkeyPointIdx)

             
        plotImg = np.empty((max(prevImg.shape[0], newImage.shape[0]), prevImg.shape[1]+newImage.shape[1], 3), dtype=np.uint8)    
      
        print(len(finalizedMatches))
        plotImg = cv2.drawMatches(prevImg,prevKps,newImage,newKps,finalizedMatches,plotImg,flags = 2)

        #rotate newImage
        (h,w) = plotImg.shape[:2]
        center = (w / 2, h /2)
        M = cv2.getRotationMatrix2D(center, 90, 1)
        plotImg = cv2.warpAffine(plotImg,M,(w,h))

        #plot
        plt.imshow(plotImg)
        plt.pause(.001)
        plt.draw()
        plt.clf()

    if idx % N_SKIP == 0:
        prevImg = newImage
        (prevKps,prevDescs) = (newKps,newDescs)


