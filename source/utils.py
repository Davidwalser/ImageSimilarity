import os
import config
from pathlib import Path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def pairwise(t):
    it = iter(t)
    return list(zip(it,it))

def load_and_resize_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder,filename),  target_size=(config.IMG_WIDTH, config.IMG_HEIGHT))
        img = img_to_array(img)
        img = img.reshape(img.shape)
        if img is not None:
            images.append(img)
    return images

def plot_pairs(pairs):
	columns = 2
	rows = 8
	fig, axs = plt.subplots(rows, columns)
	for row in range(rows):
		pair = pairs[row]
		for column in range(columns):
			img = pair[column]
			axs[row, column].imshow(img.astype('uint8'))
			axs[row, column].axis('off')
	plt.show()

images = load_and_resize_images_from_folder(Path("./SurveyPictures/PositivePairs"))
positive_pairs = pairwise(images)
plot_pairs(positive_pairs)

images = load_and_resize_images_from_folder(Path("./SurveyPictures/NegativePairs"))
negative_pairs = pairwise(images)
plot_pairs(negative_pairs)