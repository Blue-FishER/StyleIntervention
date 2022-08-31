from tqdm import trange
from utils.batch_process import get_file_path
from boundary import predictor
from boundary import train
import numpy as np
import datetime
import os
import cv2


# 将sample文件夹下的图片进行分割,row表示一张图片中一共有的子图片数量
def processImg(img_path, row=2):
    img = cv2.imread(img_path)
    img = cv2.resize(img, [1024, 1024 * row])

    time_str = datetime.datetime.now().strftime('%m_%d_%H_%M_')
    base_path = f"boundary/images/{time_str}"
    paths = []
    for i in range(row):
        path = f'{base_path}_{i}.png'
        paths.append(path)
        cv2.imwrite(path, img[i * 1024:(i + 1) * 1024, :, :])

    return paths


def detect_test():
    path = processImg("sample/07_27_06_17/000001.png")
    resp = predictor.detect(path[0])
    resp = predictor.detect("boundary/images/07_30_14_51__1.png")
    print(resp)
    score = resp["faces"][0]["attributes"]["smile"]["value"] - resp["faces"][0]["attributes"]["smile"]["threshold"]
    print(score)


def train_test():
    img_dir = "boundary/images"
    out_name = "test1"
    codes = np.random.randn(4, 512)
    scores = np.load(f"{img_dir}/{out_name}.npy")
    train.train_boundary(codes, scores, chosen_num_or_ratio=1)


def score(dir_path, batch_num):
    """
    对文件夹下的所有图片按批次进行打分
    :param dir_path: 图片所在文件夹
    :param batch_num: 批次的数目
    :return: None
    """
    score_path = f"{dir_path}/scores"
    os.mkdir(score_path)
    for i in trange(batch_num):
        scores = predictor.get_score(get_file_path(dir_path, i))
        np.save(f"{score_path}/{i:03d}", scores)

if __name__ == "__main__":
    train_test()
