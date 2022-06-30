import os, random
from pathlib import Path
from re import X
import utils
import shutil

def rename_pairs(prefix, folder):
    files = list(Path(folder).rglob("*.gif"))
    pairs = utils.pairwise(files)
    count = 1
    for img1, img2 in pairs:
        print(str(count) + ': ' + str(img1) + ' ' + str(img2))
        os.rename(img1, folder + '/' + prefix + '_' + str(count) + '_1.gif')
        os.rename(img2, folder + '/' + prefix + '_' + str(count) + '_2.gif')
        count += 1

def make_random_negative_pairs(sourceFolder, destinationFolder, count):
    files = list(Path(sourceFolder).rglob("*.gif"))
    existing_pairs_count = int(len(list(Path(destinationFolder).rglob("*.gif"))) / 2)
    print('exisitng pairs: ' + str(existing_pairs_count))
    new_index_pair = existing_pairs_count + 1
    for x in range(new_index_pair, new_index_pair + count):
        print(x)
        file1 = random.choice(files)
        file2 = random.choice(files)
        if(file1 != file2):
            print(file1)
            print(destinationFolder + '\\neg_' + str(x) + '_1.gif')
            print(file2)
            print(destinationFolder + '\\neg_' + str(x) + '_2.gif')
            print('----')
            shutil.copyfile(file1, destinationFolder + '\\neg_' + str(x) + '_1.gif')
            shutil.copyfile(file2, destinationFolder + '\\neg_' + str(x) + '_2.gif')

# make_random_negative_pairs("C:\Study\FH_Campus\MasterThesis\source\Images", "C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\NegativePairs", 300)
rename_pairs('test', './SurveyPictures/Random_Test')