import numpy as np
import cv2
from matplotlib import pyplot as plt
from requests import get

def calculateMatches(des1,des2):
    bf = cv2.BFMatcher()
    if des1 is not None and des2 is not None:
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

def get_sift_similarity_score(filename1, filename2, plot_result = False):
    try:
        img1=cv2.imread(str(filename1),cv2.IMREAD_ANYCOLOR)
        img2=cv2.imread(str(filename2),cv2.IMREAD_ANYCOLOR)

        sift=cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        good_matches = calculateMatches(des1,des2)

        if(plot_result):
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=2)
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)),plt.show()

        if good_matches is None:
            return 0
        score = 100 * (len(good_matches)/min(len(kp1),len(kp2)))
        return score
    except:
        return 0

# get_sift_similarity_score("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\test.gif", 
# "C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\test3.gif",
# True)

# get_sift_similarity_score("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\sift_test1.gif", 
# "C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\sift_test2.gif",
# True)

# get_sift_similarity_score("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\pos_148_1.gif", 
# "C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\pos_148_2.gif",
# True)