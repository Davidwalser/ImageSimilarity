# import the necessary packages
import os
# specify the shape of the inputs for our network
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 10

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])