# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:43:34 2018

@author: jonas
"""

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "wt")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_train.csv",100)# 60000)