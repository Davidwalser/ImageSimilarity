from PIL import Image as PILImage
import utils
from pathlib import Path

PILImage.MAX_IMAGE_PIXELS = None
filenames = list(Path('C:\Study\FH_Campus\MasterThesis\source\Images').rglob("*.gif"))
average_imagesize = utils.get_average_imagesize(filenames)
print('average image width: ' + str(average_imagesize[0]))
print('average image height: ' + str(average_imagesize[1]))
smallest, largest = utils.get_largest_and_smallest_image(filenames)
print('total count: ')
print(len(filenames))
print('smallest: ')
print(smallest[0])
print(smallest[1])
print('largest: ')
print(largest[0])
print(largest[1])