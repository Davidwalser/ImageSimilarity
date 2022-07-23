import os, random
from pathlib import Path
from re import X
import cv2
import utils
import shutil
import sift
import config

def rename_pairs(prefix, folder):
    files = list(Path(folder).rglob("*.gif"))
    pairs = utils.pairwise(files)
    count = 1
    for img1, img2 in pairs:
        print(str(count) + ': ' + str(img1) + ' ' + str(img2))
        os.rename(img1, folder + '/' + prefix + '_' + str(count) + '_1.gif')
        os.rename(img2, folder + '/' + prefix + '_' + str(count) + '_2.gif')
        count += 1

def generate_random_negative_pairs(sourceFolder, destinationFolder, count):
    files = list(Path(sourceFolder).rglob("*.gif"))
    existing_pairs_count = int(len(list(Path(destinationFolder).rglob("*.gif"))) / 2)
    print('[INFO] exisitng pairs: ' + str(existing_pairs_count))
    new_index_pair = existing_pairs_count + 1
    for x in range(new_index_pair, new_index_pair + count):
        file1 = random.choice(files)
        file2 = random.choice(files)
        if(file1 != file2):
            shutil.copyfile(file1, destinationFolder + '\\neg_' + str(x) + '_1.gif')
            shutil.copyfile(file2, destinationFolder + '\\neg_' + str(x) + '_2.gif')

def is_positive(filename1, filename2):
        score = sift.get_sift_similarity_score(filename1,filename2)
        if(score > config.SIFT_SIMILARITY_SCORE):
            print("[INFO] found match!")
            print("[INFO] file1: ", filename1)
            print("[INFO] file2: ",filename2)
            return True
        return False

def generate_random_positive_pairs(sourceFolder, destinationFolder, count):
    files = list(Path(sourceFolder).rglob("*.gif"))
    existing_pairs_count = int(len(list(Path(destinationFolder).rglob("*.gif"))) / 2)
    print('[INFO] exisitng pairs: ' + str(existing_pairs_count))
    new_index_pair = existing_pairs_count + 1
    new_pairs_count = 0
    while(new_pairs_count < count):
        index_to_check = random.randint(1,len(files)-2)
        file_to_check = files[index_to_check]
        file_neighbour1 = files[index_to_check-1]
        if(is_positive(file_to_check, file_neighbour1)):
            shutil.copyfile(file_to_check, destinationFolder + '\\pos_' + str(new_index_pair) + '_1.gif')
            shutil.copyfile(file_neighbour1, destinationFolder + '\\pos_' + str(new_index_pair) + '_2.gif')
            new_index_pair += 1
            new_pairs_count += 1
            print('[INFO] currently generated pairs: ', new_pairs_count)
        file_neighbour2 = files[index_to_check+1]
        if(is_positive(file_to_check, file_neighbour2)):
            shutil.copyfile(file_to_check, destinationFolder + '\\pos_' + str(new_index_pair) + '_1.gif')
            shutil.copyfile(file_neighbour2, destinationFolder + '\\pos_' + str(new_index_pair) + '_2.gif')
            new_index_pair += 1
            new_pairs_count += 1
            print('[INFO] currently generated pairs: ', new_pairs_count)
        # for x in range(100):
        #     file2 = random.choice(files)
        #     if(file_to_check != file2):
        #         if(is_positive(file_to_check, file_neighbour2)):
        #             print('pos random')
        #             new_positive_pairs.append([file_to_check, file_neighbour2])

def sift_test_pairs(folderPath):
    images = list(Path(folderPath).rglob("*.gif"))
    image_pairs = utils.pairwise(images)
    result = []
    count = 1
    print("[INFO] total pairs count: ", len(image_pairs))
    for pair in image_pairs:
        img1, img2 = pair
        similarity = sift.get_sift_similarity_score(img1,img2)
        result.append([similarity,pair])
        print("[INFO] currently checked: ", count)
        count += 1
    highest_results = sorted(result, key = lambda kv: kv[0], reverse=True)
    lowest_results = sorted(result, key = lambda kv: kv[0], reverse=False)
    plot_sift_test_pairs(highest_results)
    plot_sift_test_pairs(lowest_results)

def plot_sift_test_pairs(sorted_list):
    pair_names = [row[1] for row in sorted_list]
    pairs = []
    for img1_name, img2_name in pair_names:
        img1=cv2.imread(str(img1_name), cv2.COLOR_BGR2RGB)
        img2=cv2.imread(str(img2_name), cv2.COLOR_BGR2RGB)
        pairs.append([img1, img2])
    similarities = [row[0] for row in sorted_list]
    utils.plot_pairs(pairs, similarities)


# sift_test_pairs("C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\PositivePairs_small")
# generate_random_negative_pairs("C:\Study\FH_Campus\MasterThesis\source\Images", "C:\Study\FH_Campus\MasterThesis\source\SurveyPictures\\NegativePairs", 400)
# rename_pairs('pos', './SurveyPictures/PositivePairs')
generate_random_positive_pairs("C:\Study\FH_Campus\MasterThesis\source\Images", './SurveyPictures/PositivePairs_small', 1)