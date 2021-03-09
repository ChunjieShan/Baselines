#!/usr/bin/python3
# -*- coding: utf8 -*-

import os
import sys
import shutil
import cv2
import random


class GeneDataset():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sub_dirs = []
        self.sub_dir_images = []
        self.num_classes = 0
        self.lines = []

    def look_sub_dir(self):
        list_dir = os.walk(self.root_dir)
        for root, dirs, files in list_dir:
            for d in dirs:
                self.sub_dirs.append(os.path.join(root, d))
                print("Sub_dir = ", os.path.join(root, d))
                self.num_classes = self.num_classes + 1
                print("Num classes = ", self.num_classes)

    def reformat(self):
        label = 0
        for sub_dirs in self.sub_dirs:
            list_dir = os.walk(sub_dirs)
            for root, dirs, files in list_dir:
                for f in files:
                    src_name = os.path.join(root, f)
                    print("Src name: ", src_name)
                    src_format = src_name.split('.')[-1]
                    print("Src format is {} format".format(src_format))

                    if src_format != "jpg":
                        img = cv2.imread(src_name)
                        new_name = src_name.replace(src_format, "jpg")
                        print("New name: ", new_name)
                        cv2.imwrite(new_name, img)
                        self.lines.append(new_name + ' ' + str(label) + '\n')
                        os.remove(src_name)
                    else:
                        self.lines.append(src_name + ' ' + str(label) + '\n')
            label = label + 1

    def split_train_val(self, train_file, test_file):
        if len(self.lines):
            random.shuffle(self.lines)
            ftrain_file = open(train_file, 'w')
            ftest_file = open(test_file, 'w')
            train_length = int(0.7 * len(self.lines))
            for i in range(0, train_length):
                ftrain_file.write(self.lines[i])

            for i in range(train_length, len(self.lines)):
                ftest_file.write(self.lines[i])


if __name__ == "__main__":
    my_class_dataset = GeneDataset(sys.argv[1])
    my_class_dataset.look_sub_dir()
    my_class_dataset.reformat()
    my_class_dataset.split_train_val('train_shuffle.txt', 'val_shuffle.txt')
