import os
import glob

ROOT_DIR: str = "C:\\Users\\behna\\OneDrive\\Dokumente\\My Doccuments\\Pycharm_projects\\CARLA\src\data\\simulation2"

imgs_dir = os.path.join(ROOT_DIR, "rgb_cam")
labels_dir = os.path.join(ROOT_DIR, "sem_cam")

img_paths = glob.glob(os.path.join(imgs_dir, "*.jpg"))
label_paths = glob.glob(os.path.join(labels_dir, "*.jpg"))

# Check if there is any img_path in img_paths that does not have any corresponding label_path in label_path.
for img_path in img_paths:
    label_path = os.path.join(labels_dir, os.path.basename(img_path))
    if os.path.exists(label_path) is False:
        print(f"Label path {label_path} not found.")
        os.remove(img_path)

# Check if there is any label_path in label_paths that does not have any corresponding img_path in img_path.
for label_path in label_paths:
    img_path = os.path.join(imgs_dir, os.path.basename(label_path))
    if os.path.exists(img_path) is False:
        print(f"Image path {img_path} not found.")
        os.remove(label_path)