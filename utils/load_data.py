# 从指定文件中加载相应的数据
import cv2
from utils.batch_process import get_file_path
import numpy as np


def load_score(dir_path, batch_num):
    """
        加载文件夹中所有的记录score的文件
        :param dir_path: scores所在文件夹
        :param batch_num: 批次的数目
        :return: None
    """
    scores = np.empty([0, 1])
    for i in range(batch_num):
        score_path = get_file_path(dir_path, i, "npy")
        score = np.load(score_path[0])
        scores = np.concatenate((scores, score), axis=0)

    return scores


def load_codes(dir_path, batch_num, codes_sufix):
    """
        加载文件夹中所有的记录codes的文件
        :param dir_path: codes所在文件夹
        :param batch_num: 批次的数目
        :param codes_sufix: codes文件的后缀，目的是用于S空间的style、torgb类型进行区分，包括_st.npy _s.npy, _t.npy
        :return: None
    """
    # 先加载出第一个批次的codes，用于拼接
    code_path = get_file_path(dir_path, 0, codes_sufix)
    codes = np.load(code_path[0])

    for i in range(1, batch_num):
        code_path = get_file_path(dir_path, i, codes_sufix)
        code = np.load(code_path[0])
        codes = np.concatenate((codes, code), axis=0)

    return codes


def load_masks(dir_path, batch_num):
    """
        加载文件夹中所有图片
        :param dir_path: mask所在文件夹
        :param batch_num: 批次的数目
        :return: None
    """
    masks = np.empty(shape=[0, 3, 1024, 1024])
    for i in range(batch_num):
        paths = get_file_path(dir_path, i)
        for path in paths:
            img = cv2.imread(path)
            img[img == 255] = 1
            masks = np.concatenate((masks, img[None, ...]), axis=0)
    return masks
