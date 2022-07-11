import os
import config
from pathlib import Path
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import siamese_network
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import Image as PILImage

def pairwise(t):
	it = iter(t)
	return list(zip(it,it))

def load_and_resize_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = load_img(os.path.join(folder,filename), grayscale=False, color_mode='rgb', target_size=(config.IMG_WIDTH, config.IMG_HEIGHT))
		img = img_to_array(img)
		img = img.reshape(img.shape)
		img=img/255.
		# img = np.expand_dims(img, axis=-1)
		# np.expand_dims(img1.image_array, axis=0)
		if img is not None:
			images.append(Image(img,filename))
	return images

def plot_pairs(pairs, titles, rows = 8):
	columns = 2
	fig, axs = plt.subplots(rows, columns)
	for row in range(rows):
		pair = pairs[row]
		title = titles[row]
		for column in range(columns):
			img = pair[column]
			axs[row, column].imshow(img.astype('uint8'))
			axs[row, column].axis('off')
			axs[row, column].set_title(title)
	plt.show()
	plt.savefig(config.SIFT_PLOT_PATH)

def plot_imagepairs(pairs, titles, rows = 8):
	columns = 2
	fig, axs = plt.subplots(rows, columns)
	for row in range(rows):
		pair = pairs[row]
		title = titles[row]
		for column in range(columns):
			img = pair[column]
			axs[row, column].imshow(img.image_array.astype('uint8'))
			axs[row, column].axis('off')
			axs[row, column].set_title(title)
	plt.show()


def make_test_and_train_pairs(positive_path, negative_path, test_size):
	images = load_and_resize_images_from_folder(positive_path)
	image_arrays = (image.image_array for image in images)
	positive_pairs = pairwise(image_arrays)
	# plot_pairs(positive_pairs)

	images = load_and_resize_images_from_folder(negative_path)
	image_arrays = (image.image_array for image in images)
	negative_pairs = pairwise(image_arrays)
	# plot_pairs(negative_pairs)
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	labels = []

	for pair in positive_pairs:
		pairImages.append(pair)
		labels.append([1])
	for pair in negative_pairs:
		pairImages.append(pair)
		labels.append([0])
	
	# combined_lists = list(zip(pairImages, labels))
	# random.shuffle(combined_lists)
	# pairImages_shuffled, labels_shuffled = zip(*combined_lists)
	images_train, images_test, labels_train, labels_test = train_test_split(pairImages, labels, test_size=test_size)
	# plot_pairs(pairImages_shuffled,pairLabels_shuffled)
	return (np.array(images_train), np.array(labels_train), np.array(images_test), np.array(labels_test))

def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

def build_model():
	print("[INFO] building siamese network...")
	imgA = Input(shape=config.IMG_SHAPE)
	imgB = Input(shape=config.IMG_SHAPE)
	featureExtractor = siamese_network.build_siamese_model(config.IMG_SHAPE)
	featsA = featureExtractor(imgA)
	featsB = featureExtractor(imgB)
	# finally, construct the siamese network
	distance = Lambda(euclidean_distance)([featsA, featsB])
	outputs = Dense(1, activation="sigmoid")(distance)
	model = Model(inputs=[imgA, imgB], outputs=outputs)
	# compile the model
	print("[INFO] compiling model...")
	# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
	# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	# model.compile(loss=contrastive_loss, optimizer="adam", metrics=["accuracy"])
	return model

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath, zoom=False):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.ylim(bottom=-0.5)
	if(zoom):
		plt.ylim(top=1)
	plt.savefig(plotPath)

def plot_featuremaps(image):
	model = build_model()
	model.load_weights(config.WEIGHTS_PATH)
	for i in range(len(model.layers)):
		layer = model.layers[i]
		# check for convolutional layer
		# if 'conv' not in layer.name:
		# 	continue
		# summarize output shape
		print(i, layer.name, layer.output.shape)
	# redefine model to output right after the first hidden layer
	# layers = model.layers
	# inputs = model.inputs
	# cloned_model = clone_model(model)
	model_input_1 = Model(inputs=model.inputs, outputs=model.layers[0].output)
	# model_input_2 = Model(inputs=inputs, outputs=model.layers[1].output)
	feature_maps = model_input_1.predict(image)
	# plot all 64 maps in an 8x8 squares
	square = 1
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	plt.show()
	# fig = plt.figure(figsize=(20,15))
	# for i in range(1,features.shape[0]+1):
	# 	plt.subplot(8,8,i)
	# 	# plt.imshow(features[0,:,:,i-1] , cmap='gray')
	# 	plt.imshow(features[0,i-1] , cmap='gray')
	# plt.show()

def get_largest_and_smallest_image(filenames):
	all_images = []
	for filename in filenames:
		img = PILImage.open(filename, 'r')
		all_images.append([img.filename, img.size])
	sort = sorted(all_images, key=lambda x:x[1])
	return [sort[0],sort[-1]]
    # largest = max()
    # smallest = min(PILImage.open(f, 'r').size for f in filenames)
    # return [smallest, largest]

def get_average_imagesize(filenames):
	widths = []
	heights = []
	for filename in filenames:
		size = PILImage.open(filename, 'r').size
		widths.append(size[0])
		heights.append(size[1])
	average_width = sum(widths)/len(widths)
	average_height = sum(heights)/len(heights)
	return [average_width, average_height]

class Image:
	def __init__(self, image_array, filename):
		self.image_array = image_array
		self.filename = filename