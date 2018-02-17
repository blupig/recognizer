import os
import shutil

base_path = 'tiny-imagenet-200/val'

with open(os.path.join(base_path, 'val_annotations.txt')) as f:
    lines = f.readlines()
    file_count = 0
    for line in lines:
        file_count += 1
        print('Processing file', file_count)

        img_info = line.split('\t')
        file_name = img_info[0]
        file_class = img_info[1]

        img_directory = os.path.join(base_path, file_class)

        # Make directory
        if not os.path.exists(img_directory):
            os.mkdir(img_directory)

        # Copy file
        shutil.copy(os.path.join(base_path, 'images', file_name),
                    os.path.join(img_directory, file_name))
