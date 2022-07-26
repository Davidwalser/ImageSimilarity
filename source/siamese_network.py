import os
import config
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils.vis_utils import plot_model

def build_cnn(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(shape=inputShape)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(5, 5))(x)
	x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)
	
	# x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
	# x = MaxPooling2D(pool_size=2)(x)
	# x = Dropout(0.3)(x)
	
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	model = Model(inputs, outputs)
	model.summary()
	plot_model(model, to_file=os.path.sep.join([config.BASE_OUTPUT,'model_cnn.png']))
	# return the model to the calling function
	return model