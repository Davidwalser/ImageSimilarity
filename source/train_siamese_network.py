# import the necessary packages
from siamese_network import build_siamese_model
import config
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
from pathlib import Path



# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain, pairTest, labelTest) = utils.make_test_and_train_pairs(Path("./SurveyPictures/PositivePairs_train"),Path("./SurveyPictures/NegativePairs_train"), config.TEST_SPLIT)
# (pairTest, labelTest) = utils.make_pairs(Path("./SurveyPictures/PositivePairs_test"),Path("./SurveyPictures/NegativePairs_test"))
# configure the siamese network
print('pairs train size: '+ str(len(pairTrain)))
print('pairs test size: '+ str(len(pairTest)))
print("[INFO] building siamese network...")
model = utils.build_model()
# compile the model
# print("[INFO] compiling model...")
# # optimizer = keras.optimizers.Adam(lr=0.001)
# # model.compile(loss=utils.contrastive_loss, optimizer="adam", metrics=["accuracy"])
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# train the model
print(pairTrain[0].shape)
print(pairTrain[:, 0].shape)
print(pairTrain[:, 1].shape)
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE,
	# shuffle=True,
	epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
# model.save(config.MODEL_PATH)
model.save_weights(config.WEIGHTS_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)