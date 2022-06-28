import os
import config
from pathlib import Path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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

def pairwise(t):
	it = iter(t)
	return list(zip(it,it))

def load_and_resize_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = load_img(os.path.join(folder,filename),  target_size=(config.IMG_WIDTH, config.IMG_HEIGHT))
		img = img_to_array(img)
		img = img.reshape(img.shape)
		# img = np.expand_dims(img, axis=-1)
		# np.expand_dims(img1.image_array, axis=0)
		if img is not None:
			images.append(Image(img,filename))
	return images

def plot_pairs(pairs, titles):
	columns = 2
	rows = 8
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

def plot_training(H, plotPath):
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
	# plt.ylim(bottom=-1)
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

def make_pairs_old(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]
		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

class Image:
	def __init__(self, image_array, filename):
		self.image_array = image_array
		self.filename = filename