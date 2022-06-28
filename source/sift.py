import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename

filename1 = askopenfilename(filetypes=[("image","*.gif")]) # queryImage
filename2 = askopenfilename(filetypes=[("image","*.gif")]) # trainImage

img1=cv2.imread(filename1,4)
img2=cv2.imread(filename2,4)

# Initiate SURF detector
sift=cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SURF
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#         a=len(good)
#         percent=(a*100)/len(kp2)
#         print("{} % similarity".format(percent))
#         if percent >= 75.00:
#             print('Match Found')
#         if percent < 75.00:
#             print('Match not Found')
bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults
good_matches = calculateMatches(des1,des2)
score = 100 * (len(good_matches)/min(len(kp1),len(kp2)))
print(score)

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# plt.imshow(img3),plt.show()