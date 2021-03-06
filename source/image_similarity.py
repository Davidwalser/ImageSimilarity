from cgi import test
import config
import utils
from pathlib import Path
import numpy as np

print("[INFO] loading siamese model...")
model = utils.build_model()
model.load_weights(config.WEIGHTS_PATH)
original_images = utils.load_images(Path("./SurveyPictures/Random_Test"))
original_image_pairs = utils.pairwise(original_images)
images = utils.load_and_resize_images_from_folder(Path("./SurveyPictures/Random_Test"))
image_pairs = utils.pairwise(images)
images = np.array(image_pairs)
similarities = []
for img1, img2 in images:
	imageA = np.expand_dims(img1.image_array, axis=0)
	imageB = np.expand_dims(img2.image_array, axis=0)
	prediction = model.predict([imageA, imageB])
	# utils.plot_featuremaps([imageA, imageB])
	similarities.append(prediction[0][0])
	print("Similarity for {} and {} :  {:.2f}".format(img1.filename, img2.filename, prediction[0][0]))
utils.plot_imagepairs(original_image_pairs, similarities)