from ctypes import util
import config
import sift
import utils
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def get_similar_images(testimage, path_all_images, count):
    all_images = list(Path(path_all_images).rglob("*.gif"))
    dictionary = {}
    for image in all_images:
        similarity = sift.get_sift_similarity_score(testimage,image)
        dictionary[image] = similarity
        print(similarity)
    sorted_dictionary = dict(sorted(dictionary.items(), key = lambda kv: kv[1],reverse=True)[:count])
    print(sorted_dictionary)
    test_image=cv2.imread(str(testimage), cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)
    plt.show()
    pairs = []
    scores = []
    for key in sorted_dictionary:
        print(key, '->', sorted_dictionary[key])
        img=cv2.imread(str(key), cv2.COLOR_BGR2RGB)
        pairs.append([test_image,img])
        scores.append(sorted_dictionary[key])
    utils.plot_pairs(pairs,scores,count)
    a = 1
get_similar_images("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\test.gif","C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\Sift_test",5)