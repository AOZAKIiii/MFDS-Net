import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import sys
import random

__all__ = ['SirstAugDataset', 'MDFADataset', 'MergedDataset', 'BUSIdataset', 'DSIFNDataset']


class MergedDataset(Data.Dataset):
    def __init__(self, mdfa_base_dir='../data/MDFA', sirstaug_base_dir='../data/sirstaug', mode='train', base_size=256):
        assert mode in ['train', 'test']

        self.sirstaug = SirstAugDataset(base_dir=sirstaug_base_dir, mode=mode)
        self.mdfa = MDFADataset(base_dir=mdfa_base_dir, mode=mode, base_size=base_size)

    def __getitem__(self, i):
        if i < self.mdfa.__len__():
            return self.mdfa.__getitem__(i)
        else:
            inx = i - self.mdfa.__len__()
            return self.sirstaug.__getitem__(inx)

    def __len__(self):
        return self.sirstaug.__len__() + self.mdfa.__len__()


class MDFADataset(Data.Dataset):
    def __init__(self, base_dir='../data/MDFA', mode='train', base_size=256):
        assert mode in ['train', 'test']

        self.mode = mode
        if mode == 'train':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
        else:
            raise NotImplementedError

        self.img_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, i):
        if self.mode == 'train':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img, mask = self.img_transform(img), self.mask_transform(mask)
        return img, mask

    def __len__(self):
        if self.mode == 'train':
            return 9978
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError


class SirstAugDataset(Data.Dataset):
    def __init__(self, base_dir='./data/JSRT256', mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = os.path.join(base_dir, 'train')

        elif mode == 'test':
            self.data_dir = os.path.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []


        for filename in os.listdir(os.path.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        # print("-----")
        # print(self.names)
        # print("-----")

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        # img_h,img_w = images.size
        # if img_h!=256 and img_w!=256:
        #     images = images.resize((256,256))
        #     mask = mask.resize((256,256))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        # print(name)
        # print("---------")
        # print(name)
        # print(images.shape)
        # print(mask.shape)
        # print("----------")
        return img, mask, name

    def __len__(self):
        return len(self.names)


class BUSIdataset(Data.Dataset):
    def __init__(self, base_dir='./data/BUSI', mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = os.path.join(base_dir, 'train')

        elif mode == 'test':
            self.data_dir = os.path.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []


        for filename in os.listdir(os.path.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        # print("-----")
        # print(self.names)
        # print("-----")

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        img_h,img_w = img.size
        if img_h!=256 and img_w!=256:
            img = img.resize((256,256))
            mask = mask.resize((256,256))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        # print(name)
        # print("---------")
        # print(name)
        # print(images.shape)
        # print(mask.shape)
        # print("----------")
        return img, mask, name

    def __len__(self):
        return len(self.names)



class DSIFNDataset(Data.Dataset):
    def __init__(self, base_dir='./YGdata/LEVIR', mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = os.path.join(base_dir, 'train')

        elif mode == 'test':
            self.data_dir = os.path.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []


        for filename in os.listdir(os.path.join(self.data_dir, 'Aimages')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        # print("-----")
        # print(self.names)
        # print("-----")

    def __getitem__(self, i):
        name = self.names[i]
        Aimg_path = osp.join(self.data_dir, 'Aimages', name)
        Bimg_path = osp.join(self.data_dir, 'Bimages', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        Aimg = Image.open(Aimg_path).convert('RGB')
        Bimg = Image.open(Bimg_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        # img_h,img_w = images.size
        # if img_h!=256 and img_w!=256:
        #     images = images.resize((256,256))
        #     mask = mask.resize((256,256))

        Aimg, Bimg, mask = self.transform(Aimg), self.transform(Bimg), transforms.ToTensor()(mask)
        # print(name)
        # print("---------")
        # print(name)
        # print(images.shape)
        # print(mask.shape)
        # print("----------")
        return Aimg, Bimg, mask, name

    def __len__(self):
        return len(self.names)