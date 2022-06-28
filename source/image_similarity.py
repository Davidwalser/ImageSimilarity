from cgi import test
import config
import utils
from pathlib import Path
import numpy as np

print("[INFO] loading siamese model...")
model = utils.build_model()
model.load_weights(config.WEIGHTS_PATH)
images = utils.load_and_resize_images_from_folder(Path("./SurveyPictures/Random_Test"))
image_pairs = utils.pairwise(images)
images = np.array(image_pairs)
similarities = []
for img1, img2 in images:
	imageA = np.expand_dims(img1.image_array, axis=0)
	imageB = np.expand_dims(img2.image_array, axis=0)
	pred = model.predict([imageA, imageB])
	# utils.plot_featuremaps([imageA, imageB])
	similarities.append(pred[0][0])
	print("Similarity for {} and {} :  {:.2f}".format(img1.filename, img2.filename, pred[0][0]))
utils.plot_imagepairs(image_pairs, similarities)