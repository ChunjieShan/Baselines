#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import numpy as np
import cv2

src = cv2.imread("./4.jpg")
image = cv2.resize(src, (150, 150))
image = np.float32(image) / 255.0
image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

image = image.transpose((2, 0, 1))
input_x = torch.from_numpy(image).unsqueeze(0)
print(input_x.size())
model = torch.load("./models/model.ckpt")
pred = model(input_x.cuda())
