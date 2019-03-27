import numpy as np
import os


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folder, files in os.walk(file_dir):
        for name in sub_folder:
            temp.append(os.path.join(root, name))
        print(files)

    labels = []
    for on_folder in temp:
        n_img = len(os.listdir(on_folder))
        letter = on_folder.split("/")[-1]

        if letter == "cat":
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])

    temp = np.append([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    labels_list = list(temp[:, 1])
    labels_list = [int(float(i)) for i in labels_list]

    return image_list, labels_list
