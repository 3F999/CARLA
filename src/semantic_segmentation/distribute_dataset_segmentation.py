import os
import glob
from shutil import copy
from typing import List
import cv2
from tqdm import tqdm

ROOT_DIR = 'test'
RGB_IMAGES_DIRECTORY = 'C:\\Users\\behna\\OneDrive\\Dokumente\\My Doccuments\\Pycharm_project\\Efficient-Segmentation-Networks\\dataset\\camvid\\test'
ANNOT_IMAGES_DIRECTORY = 'C:\\Users\\behna\\OneDrive\\Dokumente\\My Doccuments\\Pycharm_project\\Efficient-Segmentation-Networks\\dataset\\camvid\\testannot'

rgb_img_paths: List[str] = glob.glob(os.path.join(ROOT_DIR, "*.jpg"))


for rgb_img_path in tqdm(rgb_img_paths):
    rgb_img_raw_name = os.path.basename(rgb_img_path)
    annotation_path = os.path.join(ROOT_DIR, rgb_img_raw_name.split('.jpg')[0] + '_mask.png')
    if os.path.exists(annotation_path) is True:
        annot_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        # copy image and annotation files
        copy(rgb_img_path, RGB_IMAGES_DIRECTORY)
        annot_destination = os.path.join(ANNOT_IMAGES_DIRECTORY, rgb_img_raw_name)
        cv2.imwrite(annot_destination, annot_img)
