from ctypes import util
import config
import sift
import utils
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def get_similar_images(testImage_path, testFolder_path, similar_count):
    all_images = list(Path(testFolder_path).rglob("*.gif"))
    image_similarity_dictionary = {}
    print("images to check: " + str(len(all_images)))
    count = 1
    for image in all_images:
        print(count)
        count += 1
        similarity = sift.get_sift_similarity_score(testImage_path,image)
        image_similarity_dictionary[image] = similarity
    sorted_image_similarity_dictionary = dict(sorted(image_similarity_dictionary.items(), key = lambda kv: kv[1],reverse=True)[:similar_count])
    test_image=cv2.imread(str(testImage_path), cv2.COLOR_BGR2RGB)
    pairs = []
    scores = []
    for image_name in sorted_image_similarity_dictionary:
        print(image_name, '->', sorted_image_similarity_dictionary[image_name])
        img=cv2.imread(str(image_name), cv2.COLOR_BGR2RGB)
        pairs.append([test_image,img])
        scores.append(sorted_image_similarity_dictionary[image_name])
    utils.plot_pairs(pairs,scores,similar_count)

get_similar_images("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\test.gif","C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\Sift_test",10)