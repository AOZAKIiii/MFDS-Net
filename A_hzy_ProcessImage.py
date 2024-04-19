# -*- codeing = utf-8 -*-
import cv2
from PIL import Image
import os
import re

def image_rename(img_path, mask_path):
    print("----image_rename process----")

    img_path  = os.path.join(img_path)   
    mask_path = os.path.join(mask_path)  

    imglist   = os.listdir(img_path)
    masklist = os.listdir(mask_path)
    print("image_len:{}".format(len(imglist)))
    print("mask_len:{}".format(len(masklist)))

    for idx, path in enumerate(imglist):
        img_name = path[0:2]           
        img_name = str(img_name)+".png"

        img_name = img_path + img_name
        path = img_path + path          

        os.rename(path, img_name)

    for idx, path in enumerate(masklist):
        mask_name = path[0:2]
        mask_name = str(mask_name)+".png"

        mask_name = mask_path + mask_name
        path = mask_path + path

        os.rename(path, mask_name)

    print("----image_rename end----")


def test():
    path = "./data/DRIVE2/train/masks/"

    path = os.path.join(path)

    img_list = os.listdir(path)

    print(img_list)
    for idx, pa in enumerate(img_list):
        img_path = path + pa
        img = Image.open(img_path)
        image = cv2.imread(img_path)
        print(image.shape)


def resize_rename(img_path, mask_path, new_img_path, new_mask_path):
    if not os.path.exists(new_img_path):
        os.makedirs(new_img_path)
        print(new_img_path)

    if not os.path.exists(new_mask_path):
        os.makedirs(new_mask_path)
        print(new_mask_path)

    img_path = os.path.join(img_path)  
    mask_path = os.path.join(mask_path)  
    imglist = sorted(os.listdir(img_path))
    masklist = sorted(os.listdir(mask_path))
    print("image_len:{}".format(len(imglist)))
    print("mask_len:{}".format(len(masklist)))

    for idx, path in enumerate(imglist):
        # img_name = str(path[0:2])
        img_name = str(re.sub("\D", "", path))
        # print(img_name)

        path = img_path+path  

        img = Image.open(path)
        img = img.resize((256, 256))
        img.save(new_img_path + img_name + ".png")
        print(type(img))


    for idx, path in enumerate(masklist):
        # mask_name = str(path[0:2])
        mask_name = str(re.sub("\D", "", path))
        # print(mask_name)

        path = mask_path+path  

        img = Image.open(path)
        img = img.resize((256, 256))
        img.save(new_mask_path + mask_name + ".png")
        print(type(img))


def checktype(img_path, mask_path):
    img_path = os.path.join(img_path)  
    mask_path = os.path.join(mask_path)  

    imglist = os.listdir(img_path)
    masklist = os.listdir(mask_path)
    for idx,path in enumerate(imglist):
        img = img_path + path
        mask = mask_path + path

        image = cv2.imread(img)
        print(image.shape)
        mask = cv2.imread(mask,0)
        print(mask.shape)



if __name__ == '__main__':
    img_path  = "./data/DRIVE2/train/images/"
    mask_path = "./data/DRIVE2/train/masks/"

    new_img_path  = "./data/Corneal2/test/images/"
    new_mask_path = "./data/Corneal2/test/masks/"

    # image_rename(img_path, mask_path)
    # test()
    # resize_rename(img_path, mask_path, new_img_path, new_mask_path)
    checktype(img_path, mask_path)