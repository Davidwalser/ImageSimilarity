# import the necessary packages
from siamese_network import build_cnn
import config
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
from pathlib import Path
import tensorflow as tf

def train_siamese_network(positivePairs_path, negativePairs_path, testSplit):
	# prepare the positive and negative pairs
	print("[INFO] preparing positive and negative pairs...")
	(pairTrain, labelTrain, pairTest, labelTest) = utils.make_test_and_train_pairs(positivePairs_path, negativePairs_path, testSplit)
	print('[INFO] pairs train size: '+ str(len(pairTrain)))
	print('[INFO] pairs test size: '+ str(len(pairTest)))
	# utils.plot_pairs(pairTrain, labelTrain)

	model = utils.build_model()
	print("[INFO] compiling model...")
	# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
	# model.compile(loss=utils.contrastive_loss, optimizer="adam", metrics=["accuracy"])
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	print("[INFO] training model...")
	history = model.fit(
		[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
		validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
		batch_size=config.BATCH_SIZE,
		shuffle=True,
		epochs=config.EPOCHS)

	# serialize the model to disk
	print("[INFO] saving siamese model...")
	# model.save(config.MODEL_PATH)
	model.save_weights(config.WEIGHTS_PATH)
	# plot the training history
	print("[INFO] plotting training history...")
	utils.plot_training(history, config.PLOT_PATH)
	utils.plot_training(history, config.PLOT_PATH_ZOOM, True)

train_siamese_network(Path("./SurveyPictures/PositivePairs_small"),Path("./SurveyPictures/NegativePairs_small"), config.TEST_SPLIT)