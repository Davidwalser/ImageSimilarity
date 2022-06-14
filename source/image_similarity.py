from cgi import test
import config
import utils
from pathlib import Path
import numpy as np

print("[INFO] loading siamese model...")
model = utils.build_model()
model.load_weights(config.WEIGHTS_PATH)
# model = load_model(config.MODEL_PATH)
images = utils.load_and_resize_images_from_folder(Path("./SurveyPictures/Random_Test"))
image_pairs = utils.pairwise(images)
pairImages = []
for pair in image_pairs:
	pairImages.append(pair)
images = np.array(pairImages)
# test_images = utils.load_and_resize_images_from_folder()
# # pred = model.predict([test_images[0], test_images[1]])
# # print("Similarity: {:.2f}".format(pred[0][0]))
# test_pairs = utils.pairwise(test_images)
# test_pairs_array = np.array(utils.pairwise(test_images))
# print(test_pairs[0].shape)

print(images[0].shape)
print(images[1].shape)
for img1, img2 in images:
	img1 = img1 / 255.0
	img2 = img2 / 255.0
	imageA = np.expand_dims(img1, axis=0)
	imageB = np.expand_dims(img2, axis=0)
	pred = model.predict([imageA, imageB])
	print("Similarity: {:.2f}".format(pred[0][0]))