import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy.ndimage import rotate
from collections import deque
import time

DIST_TRESHOLD = 0.1                     # euclidean distance threshold for the SURF descriptor match filter
N_SKIP = 2                              # gap in images between image matching reset
TEMP_SIZE_FACTOR = 2.6                  # ratio factor (template area / feature_size) for template matching 
OBJECT_SCALE_DETECTION_TH = 1.2         # scale increase detection treshold
ERROR_DECREASE_FACTOR = 0.7                   # factor by which the template matching error must have improved compared to scale 1
TEMP_MATCH_NUM_OF_SCALE_IT = 15         # number of iterations in the scaling template matching procedure 
SURF_HESSIAN_TRESHOLD = 5000            # threshold of hessian for the SURF detection -> depends on image quality
SURF_IS_UPRIGHT = True                  # Use U-surf to disregard rotation invariance for performance boost
TEMPLATE_IP_METHOD = cv2.INTER_LINEAR   # template resizing method
EDGE_DETECTOR_IS_ON = True              # Apply edge detector at template matching

prevImgQueue = deque([],N_SKIP)
prevDescsQueue = deque([],N_SKIP)
prevKpsQueue = deque([],N_SKIP)

images = [cv2.imread(file) for file in sorted(glob.glob('huistest/*.jpg'))]

for frameNum,newImg in enumerate(images):
    
    """ newImg = rotate(newImg,90) #optional, depends on footage """
    grayNewImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
    (imgHeight,imgWidth) = grayNewImg.shape[:2] 

    #obtain Surf features       
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = SURF_HESSIAN_TRESHOLD, upright = SURF_IS_UPRIGHT)       
    (newKps,newDescs) = surf.detectAndCompute(newImg,None)  
    
    #we will store objects in here
    objectKeypoints = []
    
    if frameNum > N_SKIP-1: #wait untill queue contains N_SKIP images so we can match with the oldest one

        #Unpack attributes of the oldest image in the queue to match with
        prevImg = prevImgQueue.popleft()
        prevDescs = prevDescsQueue.popleft()
        prevKps = prevKpsQueue.popleft()

        #Match with previous newImg
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)  #FLANN more efficient?
        matches = bf.match(prevDescs,newDescs) #  matches = bf.radiusMatch(prevDescs,newDescs, DIST_TRESHOLD) more efficient?

        #Filter out "worse" matches
        matches = [match for match in matches if match.distance < DIST_TRESHOLD]      
        
        #Filter out matches whose size has decreased or stayed the same
        finalizedMatches = []
        for idx in range(len(matches)): 

            prevKeyPointIdx = matches[idx].queryIdx         
            newkeyPointIdx = matches[idx].trainIdx          
            sizePrevKeyPoint = prevKps[prevKeyPointIdx].size
            sizeNewKeyPoint = newKps[newkeyPointIdx].size

            if sizeNewKeyPoint > sizePrevKeyPoint:
                finalizedMatches.append(matches[idx])                   
        
        finalNewKps = [newKps[match.trainIdx] for match in finalizedMatches] 
        finalPrevKps = [prevKps[match.queryIdx] for match in finalizedMatches]  
        
        #Confirm scale by template matching
        #First create template image from previous image
        prevSizes = [kpt.size for kpt in finalPrevKps]
        prevPointsXY = [kpt.pt for kpt in finalPrevKps]
        newPointsXY = [kpt.pt for kpt in finalNewKps]   
        
        for idx, prevXYPoint in enumerate(prevPointsXY):    

            templateLength = round(prevSizes[idx]*TEMP_SIZE_FACTOR)

            #check if desired template of previous keypoint would exceed the image size
            if (int(prevXYPoint[0])+templateLength/2 > imgWidth or int(prevXYPoint[0])-templateLength/2 < 0
                or int(prevXYPoint[1])+templateLength/2 >imgHeight or int(prevXYPoint[1])-templateLength/2 < 0):

                continue     
            else:  

                #create the template around previous keyPoint
                patchSize = (templateLength, templateLength)                
                prevTemplate = cv2.getRectSubPix(prevImg,patchSize,prevXYPoint)
                if EDGE_DETECTOR_IS_ON:     
                    prevTemplate = cv2.Canny(prevTemplate, 50, 200)

                lowestError = float('inf')
                errAtScaleOne = float('inf')
                bestMatchingScale = 1

                #Loop through template matching procedure by scaling
                for scale in np.linspace(1.0,1.5,TEMP_MATCH_NUM_OF_SCALE_IT):
                    
                    #Expand prevTemplate at scale 
                    resizedPrevTemp = cv2.resize(prevTemplate, (0,0), fx= scale, fy= scale, interpolation = TEMPLATE_IP_METHOD)

                    """ plt.subplot(1,2,1)
                    plt.gca().set_title('img = n -%i'%(N_SKIP))
                    plt.imshow(resizedPrevTemp,cmap="gray")                    
                    plt.draw()  """ 

                    # Second, create a template in the image under analysis using the matched keypoint,
                    # first we have check if desired template around the matching keypoint for current image exceeds the image size
                    matchPointXY = newPointsXY[idx]

                    # See if the desired New template does not exceed the image size and exit matching procedure if it does 
                    if (int(matchPointXY[0]+templateLength*scale/2) > imgWidth or int(matchPointXY[0]-templateLength*scale/2) < 0
                        or int(matchPointXY[1]+templateLength*scale/2) > imgHeight or int(matchPointXY[1]-templateLength*scale/2 < 0) ):
                        
                        break                 
                    
                    else:                  

                        # create the template around the matching keypoint
                        newTemplate = cv2.getRectSubPix(grayNewImg,resizedPrevTemp.shape,matchPointXY)                        
                        if EDGE_DETECTOR_IS_ON:
                            newTemplate = cv2.Canny(newTemplate, 50, 200)

                        """ plt.subplot(1,2,2)
                        plt.imshow(newTemplate,cmap="gray")
                        plt.gca().set_title('img = n')
                        plt.draw()
                        plt.pause(0.01) """ 
                        
                        #Compare them using MEAN SQUARED ERROR
                        err = np.sum((resizedPrevTemp.astype("float") - newTemplate.astype("float")) ** 2)
                        err /= float(resizedPrevTemp.shape[0] * newTemplate.shape[1])                        
                                                
                        # update best match                       
                        if err < lowestError:
                            lowestError = err
                            bestMatchingScale = scale

                        if scale == 1: # as a reference, store the matchingfactor at scale 1 which we will use later                            
                            errAtScaleOne = err                  
                             
                # Mark the keypoints in the new image frame as objects if the following condition holds
                if (bestMatchingScale >= OBJECT_SCALE_DETECTION_TH and lowestError <= ERROR_DECREASE_FACTOR*errAtScaleOne):
                    objectKeypoints.append(finalNewKps[idx])                 

        # Draw filtered keypoints (no objects yet)
        plotImg = np.empty((newImg.shape[0], newImg.shape[1], 3), dtype=np.uint8)
        cv2.drawKeypoints(newImg, finalNewKps,plotImg,(255,0,0),0)

        if len(objectKeypoints) > 0:
            # Plot a rectangle covering all obstacle keypoints
            ObjectsXY = [objectKeypoint.pt for objectKeypoint in objectKeypoints]            
            X = [int(objectXY[0]) for objectXY in ObjectsXY]  
            Y = [int(objectXY[1]) for objectXY in ObjectsXY]
            font = cv2.FONT_HERSHEY_SIMPLEX            
            for objects in list(zip(X,Y)):            
                cv2.putText(plotImg,"CAUTION OBJECT",objects,font,1,(0,0,255),2)
                cv2.circle(plotImg,objects,8,(0,0,255),-1)                            
            
        #Show figure
        plt.imshow(plotImg)
        plt.pause(.001)
        plt.draw()
        plt.clf()
      
    #Add the image, descriptors and keypoints to the queue holding N_SKIP items
    prevImgQueue.append(grayNewImg)
    prevDescsQueue.append(newDescs)
    prevKpsQueue.append(newKps)   
    
    

   

    
    
    

