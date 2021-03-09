#!/usr/bin/env bash
# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues

./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/train/glioma_tumor/ ./tmp/ glioma_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/train/pituitary_tumor/ ./tmp/ pituitary_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/train/meningioma_tumor/ ./tmp/ meningioma_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/train/no_tumor/ ./tmp/ no_tumor

./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/val/glioma_tumor/ ./tmp/ glioma_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/val/pituitary_tumor/ ./tmp/ pituitary_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/val/meningioma_tumor/ ./tmp/ meningioma_tumor
./rename_files_function.sh /home/rick/Datasets/brain_tumor_class/val/no_tumor/ ./tmp/ no_tumor
