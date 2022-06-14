import os
from pathlib import Path
import utils

def rename_pairs(prefix, folder):
    files = list(Path(folder).rglob("*.gif"))
    pairs = utils.pairwise(files)
    count = 1
    for img1, img2 in pairs:
        # labeled_image = (file.parent,cv2.imread(str(file)))
        #os.rename('guru99.txt','career.guru99.txt')
        print(str(count) + ': ' + str(img1) + ' ' + str(img2))
        os.rename(img1, folder + '/' + prefix + '_' + str(count) + '_1.gif')
        os.rename(img2, folder + '/' + prefix + '_' + str(count) + '_2.gif')
        count += 1

rename_pairs('pos', './SurveyPictures/rename')