from ctypes import util
import config
import sift
import utils
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def get_similar_images_sift(testImage_path, testFolder_path, similar_count):
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

def get_similar_images_siamese(testImage_path, testFolder_path, similar_count):
    # all_images = list(Path(testFolder_path).rglob("*.gif"))
    original_test_image = cv2.cvtColor(utils.load_images(testImage_path)[0].image_array, cv2.COLOR_BGR2RGB)
    # original_images = utils.load_images(testFolder_path)
    images = utils.load_and_resize_images_from_folder(testFolder_path)
    print("[INFO] images to check: " + str(len(images)))
    test_image = utils.load_and_resize_images_from_folder(testImage_path)[0]
    test_image_dimension = np.expand_dims(test_image.image_array, axis=0)
    print("[INFO] loading siamese model...")
    model = utils.build_model()
    model.load_weights(config.WEIGHTS_PATH)
    image_similarity_dictionary = {}
    count = 1
    for image in images:
        print(count)
        count += 1
        image_dimension = np.expand_dims(image.image_array, axis=0)
        similarity = model.predict([test_image_dimension, image_dimension])
	    # utils.plot_featuremaps([imageA, imageB])
        image_similarity_dictionary[image.filename] = similarity
    sorted_image_similarity_dictionary = dict(sorted(image_similarity_dictionary.items(), key = lambda kv: kv[1],reverse=True)[:similar_count])
    pairs = []
    scores = []
    for image_name in sorted_image_similarity_dictionary:
        print(image_name, '->', sorted_image_similarity_dictionary[image_name])
        img=cv2.imread(str(Path.joinpath(Path(testFolder_path),image_name)), cv2.COLOR_BGR2RGB)
        pairs.append([original_test_image,img])
        scores.append(sorted_image_similarity_dictionary[image_name])
    utils.plot_pairs(pairs,scores,similar_count)

# get_similar_images_sift("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\test.gif","C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\Sift_test",10)
get_similar_images_siamese("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\Test","C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\Sift_test",10)