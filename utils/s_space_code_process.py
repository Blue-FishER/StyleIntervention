# 用于S空间的codes的拼接或拆分
import torch
import numpy as np
from models.model_settings import CHANNELS


# 拼接卷积层和torgb层中的codes
def cat_style_trgb(style_codes, trgb_codes):
    # 合成第0层
    codes = np.concatenate((style_codes[0], trgb_codes[0]), axis=1)
    # 合成剩下的八层
    i = 1
    while i < 9:
        codes = np.concatenate((codes, style_codes[2 * i - 1], style_codes[2 * i], trgb_codes[i]), axis=1)
        i += 1
    return codes


def tensor_cat_style_trgb(style, trgb):
    # 合成第0层
    codes = torch.cat([style[0], trgb[0]], dim=1)
    # 合成剩下的八层
    i = 1
    while i < 9:
        codes = torch.cat([codes, style[2 * i - 1], style[2 * i], trgb[i]], dim=1)
        i += 1
    return codes


# 单独的拼接给定的code
def cat_codes(codes):
    return np.concatenate(codes, axis=1)


def tensor_cat(codes):
    return torch.cat(codes, dim=1)


def parse_codes(codes, code_type=0):
    """
    将输入的codes解析为list

    :param codes: 输入的codes
    :param code_type: 0-包含style和trgb    1-只包含style    2-只包含trgb
    :return:
    """
    style_codes = []
    trgb_codes = []
    # 起始索引
    idx = 0
    upper_idx = 0
    # 解析第一层
    if code_type == 0 or code_type == 1:
        upper_idx = idx + CHANNELS[4]
        style_codes.append(codes[:, idx:upper_idx])
        idx = upper_idx
    if code_type == 0 or code_type == 2:
        upper_idx = idx + CHANNELS[4]
        trgb_codes.append(codes[:, idx:upper_idx])
        idx = upper_idx

    # 解析剩余层
    for i in range(3, 11):
        if code_type == 0 or code_type == 1:
            upper_idx = idx + CHANNELS[2**(i-1)]
            style_codes.append(codes[:, idx:upper_idx])
            idx = upper_idx

            upper_idx = idx + CHANNELS[2 ** i]
            style_codes.append(codes[:, idx:upper_idx])
            idx = upper_idx
        if code_type == 0 or code_type == 2:
            upper_idx = idx + CHANNELS[2**i]
            trgb_codes.append(codes[:, idx:upper_idx])
            idx = upper_idx

    return [style_codes, trgb_codes]