import torch
from torch.autograd import Variable

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
from PIL import Image


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of AGPCNet')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'./Fifth-net_result/dsb2/2022-08-17_23-35-38_agpcnet_1/checkpoint/Iter- 2740_mIoU-0.8034_fmeasure-0.8910.pkl',
                        help='checkpoint path')
    parser.add_argument('--save-Ori_result-path', type=str, default=r'./Fifth-net_result/dsb2/2022-08-17_23-35-38_agpcnet_1/valphoto0/')
    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'./data/dsb2/test/images/', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')

    args = parser.parse_args()
    return args


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


if __name__ == '__main__':
    args = parse_args()

    # load network
    print('...load checkpoint: %s' % args.pkl_path)
    net = torch.load(args.pkl_path, map_location=torch.device('cpu'))
    net.eval()

    # load image
    print('...loading test image: %s' % args.image_path)
    image_path = os.path.join(args.image_path)
    image_list = sorted(os.listdir(image_path))
    print(image_list)
    for idx, name in enumerate(image_list):
        img = cv2.imread(image_path + name, 1)
        img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
        input = preprocess_image(img)

        # inference in cpu
        print('...inference in progress')
        with torch.no_grad():
            output, laylist, lay2list = net(input)

        output = output.cpu().detach().numpy().reshape(args.base_size, args.base_size)
        
        output = output > 0
        
        output = output + 0
        output = np.uint8(output)
        output[output==1]=255

        im = Image.fromarray(output)  
        save_path = args.save_result_path + name
        im.save(save_path)

        # show results
        # plt.figure()
        # plt.subplot(121), plt.imshow(images, cmap='gray'), plt.title('Original Image')
        # plt.subplot(122), plt.imshow(output, cmap='gray'), plt.title('Inference Result')
        # plt.show()