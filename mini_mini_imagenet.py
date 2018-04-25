import os
import glob
from PIL import Image
import shutil

num_classes = 5
image_size = (84, 84)

metatrain_folder = './data/miniImagenet/train'
obj_dir = './miniminiImagenet'
os.makedirs(obj_dir)

chosen_folders_local = os.listdir(metatrain_folder)[0:50]
chosen_folders = [os.path.join(metatrain_folder, label) for label in chosen_folders_local\
                  if os.path.isdir(os.path.join(metatrain_folder, label))]



for chosen_folder_local in chosen_folders_local:
    # make obj dirs
    obj_dir_folder = os.path.join(obj_dir, chosen_folder_local)
    os.makedirs(obj_dir_folder)
    for chosen_folder in chosen_folders:
        if chosen_folder_local in chosen_folder:
            print('get')
            for img in os.listdir(chosen_folder)[0:20]:
                shutil.copyfile(os.path.join(chosen_folder ,img), os.path.join(obj_dir_folder, img))












