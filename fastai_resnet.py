from fastai.vision import conv_layer, res_block, Flatten
import torch.nn as nn


def conv2(ni, nf):
    return conv_layer(ni, nf, stride=2)


def conv_and_res(ni, nf):
    return nn.Sequential(conv2(ni, nf), res_block(nf))


def resnet_model():
    return nn.Sequential(
        conv_and_res(1, 8),
        conv_and_res(8, 16),
        conv_and_res(16, 32),
        conv_and_res(32, 16),
        conv2(16, 10),
        Flatten())
