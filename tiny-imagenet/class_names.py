import os


# Build a list of class names corresponding to alphabetical order of
# directories in base_path/train
# Works with Stanford CS231N Tiny ImageNet directory structure
# https://tiny-imagenet.herokuapp.com/
def get_names(base_path):
    # Build full id-word mapping from words.txt
    full_dict = {}
    f = open(os.path.join(base_path, 'words.txt'))
    lines = f.readlines()
    for line in lines:
        pair = line.split('\t')
        full_dict[pair[0]] = pair[1].strip()

    # Match class names and add to list
    class_names = []
    for _, dirs, _ in os.walk(os.path.join(base_path, 'train')):
        dirs.sort()
        for d in dirs:
            class_names.append(full_dict[d])

        # Only first level
        break

    return class_names
