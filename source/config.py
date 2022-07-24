# import the necessary packages
import os
IMG_WIDTH = 415 # average: 830.66
IMG_HEIGHT = 252 # average: 504.985
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
TEST_SPLIT=0.2
BATCH_SIZE = 10
EPOCHS = 50
SIFT_SIMILARITY_SCORE = 9

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
SIFT_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "sift_plot.png"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
PLOT_PATH_ZOOM = os.path.sep.join([BASE_OUTPUT, "plot_zoom.png"])
WEIGHTS_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model", "weights.h5"])