# import the necessary packages
import os
# specify the shape of the inputs for our network
IMG_WIDTH = 160
IMG_HEIGHT = 160
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
# specify the batch size and number of epochs
TEST_SPLIT=0.2
BATCH_SIZE = 32
EPOCHS = 100

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
WEIGHTS_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model", "weights.h5"])