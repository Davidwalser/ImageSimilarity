# import the necessary packages
import os
# specify the shape of the inputs for our network
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
# specify the batch size and number of epochs
TEST_SPLIT=0.2
BATCH_SIZE = 32
EPOCHS = 50

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
PLOT_PATH_ZOOM = os.path.sep.join([BASE_OUTPUT, "plot_zoom.png"])
WEIGHTS_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model", "weights.h5"])