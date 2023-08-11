'''
This file serves as the entrance of program. It specifies the hyperparameters needed to train, evaluate and test
Efficient Net V2 model. User may change the setting during training via CLI command.
'''
import argparse
import os

from EfficientNetV2.Evaluate import evaluate
from EfficientNetV2.Train import train
from EfficientNetV2.demo import demo
from Utils import classes_txt

parser = argparse.ArgumentParser(description='EfficientNetV2 arguments')
parser.add_argument('--mode', dest='mode', type=str, default='demo', help='Mode of net')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='Epoch number of training') # epoahes: 50
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512, help='Value of batch size') # batch size: 512, user may decrease batch size by half if the computer run out of memory during training
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='Value of lr') # learning rate: 0.0001
parser.add_argument('--img_size', dest='img_size', type=int, default=32, help='reSize of input image') # picture size : 32x32
parser.add_argument('--data_root', dest='data_root', type=str, default='../../data/', help='Path to data')
parser.add_argument('--log_root', dest='log_root', type=str, default='../../log/', help='Path to model.pth')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=3755, help='Classes of character') # num_of_char: 3755
parser.add_argument('--demo_img', dest='demo_img', type=str, default='../asserts/pei.png', help='Path to demo image')
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.data_root + 'train.txt'):
        classes_txt(args.data_root + 'train', args.data_root + 'train.txt', args.num_classes)
    if not os.path.exists(args.data_root + 'test.txt'):
        classes_txt(args.data_root + 'test', args.data_root + 'test.txt', args.num_classes)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'demo':
        demo(args)
    else:
        print('Unknown mode')