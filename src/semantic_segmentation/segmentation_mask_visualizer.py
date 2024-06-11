import os
import glob
import cv2
import numpy as np
# import imutils
from tqdm import tqdm


label_map_dict = {
    # Set colors for 0 up to 13
    0: (0, 0, 0),  # None
    1: (50, 234, 157),  # Lanes
    2: (35, 142, 107),  # Road and vegetation
    3: (180, 130, 70),  # Sky
}

ROOT_DIR: str = ("C:\\Users\\behna\\OneDrive\\Dokumente\\My Doccuments\\"
                 "Pycharm_projects\\CARLA\src\data\\simulation2")

imgs_dir = os.path.join(ROOT_DIR, "rgb_cam")
labels_dir = os.path.join(ROOT_DIR, "sem_cam")

img_paths = glob.glob(os.path.join(imgs_dir, "*.jpg"))
label_paths = glob.glob(os.path.join(labels_dir, "*.jpg"))

for img_path in tqdm(img_paths, desc="Processing images"):
    label_path = os.path.join(labels_dir, os.path.basename(img_path))
    if os.path.exists(label_path) is True:
        img = cv2.imread(img_path)
        gray_seg_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        color_seg_mask = np.zeros((gray_seg_mask.shape[0], gray_seg_mask.shape[1], 3),
                                  dtype=np.uint8)
        for key in label_map_dict.keys():
            color_seg_mask[gray_seg_mask == key] = label_map_dict[key]
        # cv2.imshow("img", img)
        # cv2.imshow("seg_mask", color_seg_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # If the gray seg mask has any other value than the keys in the label_map_dict, we convert it to 0
        gray_seg_mask[~np.isin(gray_seg_mask, list(label_map_dict.keys()))] = 0
        cv2.imwrite(label_path, gray_seg_mask)















