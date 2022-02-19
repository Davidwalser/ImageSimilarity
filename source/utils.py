import os
import cv2
from pathlib import Path

result = list(Path("./train_and_test/train").rglob("*.jpg"))
for file in result:
    labeled_image = (file.parent,cv2.imread(str(file)))
    # print(labeled_image[0])
    print(labeled_image[1].shape)
    

# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
#             images.append(img)
#     return images

# def load_and_label_images(directory):
#     listOfFiles = list()
#     for (dirpath, dirnames, filenames) in os.walk(directory):
#         print(dirpath)
#         print(filenames)
#         listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#         for file in listOfFiles:
#             print(file)
#             print('----------')


# load_and_label_images('./train_and_test/train')