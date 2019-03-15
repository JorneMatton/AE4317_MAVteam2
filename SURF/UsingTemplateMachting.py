import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import imutils
from scipy.ndimage import rotate
from skimage.measure import compare_ssim
import matplotlib.patches as patches

DIST_TRESHOLD = 0.1             # euclidean distance threshold for the SURF descriptor match filter
N_SKIP = 3                     # gap in images between image matching reset
TEMP_SIZE_FACTOR = 1.6         # ratio factor (template area / feature_size) for template matching 
OBJECT_SCALE_DETECTION_TH = 1.1 # scale detection treshold
TEMP_MATCH_NUM_OF_SCALE_IT = 10 # number of iterations in the scaling template matching procedure 

images = [cv2.imread(file) for file in sorted(glob.glob('*.jpg'))]

for idx,newImg in enumerate(images):

    newImg = rotate(newImg,90)

    #obtain Surf features
    grayNewImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)    
    surf = cv2.xfeatures2d.SURF_create()
    '''surf.setUpright(True)   #Use U-surf to disregard rotation invariance for performance boost '''  
    (newKps,newDescs) = surf.detectAndCompute(newImg,None)  

    (imgHeight,imgWidth) = grayNewImg.shape[:2]
    objectKeypoints = [] #we will store objects in here

    
    if idx > 0: #skip only the first image since we cannot match it with anything yet

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
                                    
                maxScore = -1.1 #minimum SSI score is -1

                #Loop through template matching procedure by scaling
                for scale in np.linspace(1.0,1.5,TEMP_MATCH_NUM_OF_SCALE_IT):
                    
                    #Expand prevTemplate at scale 
                    resizedPrevTemp = cv2.resize(prevTemplate, (0,0), fx= scale, fy= scale, interpolation = cv2.INTER_CUBIC)

                    """ plt.subplot(1,2,1)
                    plt.gca().set_title('(img = n-1)')
                    plt.imshow(resizedPrevTemp,cmap="gray")                    
                    plt.draw() """  

                    # Second, create a template in the image under analysis using the matched keypoint,
                    # first we have check if desired template around the matching keypoint for current image exceeds the image size
                    matchPointXY = newPointsXY[idx]

                    # See if the desired New template does not exceed the image size
                    if (int(matchPointXY[0]+templateLength*scale/2) > imgWidth or int(matchPointXY[0]-templateLength*scale/2) < 0
                        or int(matchPointXY[1]+templateLength*scale/2) > imgHeight or int(matchPointXY[1]-templateLength*scale/2 < 0) ):
                        
                        break  #exit matching procedure                   
                    
                    else:                  

                        # create the template around the matching keypoint
                        newTemplate = cv2.getRectSubPix(grayNewImg,resizedPrevTemp.shape,matchPointXY)

                        """ plt.subplot(1,2,2)
                        plt.imshow(newTemplate,cmap="gray")
                        plt.gca().set_title('(img = n)')
                        plt.draw()
                        plt.pause(0.01) """ 
                        
                        # now match the templates and compute a matching metric using Structural similarity index
                        (score, _) = compare_ssim(resizedPrevTemp, newTemplate, full=True) #score is number between -1 and 1 [worse to best]
                        #WARNING top line might have to be changed to include scale factor^2
                        
                        if score > maxScore:

                            maxScore = score
                            bestMatchingScale = scale

                        if scale == 1: # as a reference, store the matchingfactor at scale 1 which we will use later
                            
                            (scoreAtScaleOne, _) = compare_ssim(resizedPrevTemp, newTemplate, full=True)                     

                             
                # Mark the keypoints in the new image frame as objects if the following condition holds
                if (bestMatchingScale > OBJECT_SCALE_DETECTION_TH and maxScore > scoreAtScaleOne):

                    objectKeypoints.append(finalNewKps[idx])                 

        # Draw normal keypoints
        plotImg = np.empty((newImg.shape[0], newImg.shape[1], 3), dtype=np.uint8)
        cv2.drawKeypoints(newImg, finalNewKps,plotImg,(255,0,0))

        if len(objectKeypoints) > 0:

            # Plot a rectangle covering all obstacle keypoints
            ObjectsXY = [objectKeypoint.pt for objectKeypoint in objectKeypoints]            
            X = [int(objectXY[0]) for objectXY in ObjectsXY]  
            Y = [int(objectXY[1]) for objectXY in ObjectsXY]
            font = cv2.FONT_HERSHEY_SIMPLEX            
            for objects in list(zip(X,Y)):            
                cv2.putText(plotImg,"CAUTION OBJECT",objects,font,0.3,(0,0,255),1)
                cv2.circle(plotImg,objects,4,(0,0,255),-1)                            
            
        #Show figure
        plt.imshow(plotImg)
        plt.pause(.5)
        plt.draw()
        plt.clf()

    if idx % N_SKIP == 0:
        prevImg = grayNewImg
        (prevKps,prevDescs) = (newKps,newDescs)


