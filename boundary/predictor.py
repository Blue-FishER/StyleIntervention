import os
import requests
from tqdm import *
import numpy as np


def detect(img_path):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "i21HGJ6h0ootjcifBRkVKsYtNGHTzpbQ"
    secret = "EPomqHCQhE3GVgcwiGoXMKtQNfAnXU3B"

    # 必需的参数，注意key、secret、"gender,age,smiling,beauty"均为字符串，与官网要求一致
    # return_landmark就是人脸关键点，0是默认不调用，1是83个关键点，2是106个
    # return_attributes是你需要，需要什么就调什么，内容在API文档可查
    data = {
        "api_key": key,
        "api_secret": secret,
        "return_attributes": "smiling"
    }

    # 以二进制读入图像，这个字典中open(filepath1, "rb")返回的
    # 是二进制的图像文件，所以"image_file"是二进制文件，符合官网要求
    files = {"image_file": open(img_path, "rb")}

    # post上传
    response = requests.post(http_url, data=data, files=files)
    # 输出
    response = response.json()
    return response


def get_score(paths):
    """
    对给定文件夹下的所有图片打分，并保存至output_name命名的npy文件中
    :param paths: 图片所在的路径
    :return: 给定图片的分数， 大小为[num, 1]
    """
    scores = []
    pbar = tqdm(total=len(paths))

    for path in paths:
        resp = detect(path)
        faces = resp["faces"]
        if len(faces) == 0:
            score = 0
        else:
            score = faces[0]["attributes"]["smile"]["value"] - resp["faces"][0]["attributes"]["smile"][
            "threshold"]
        scores.append(score)
        pbar.update(1)

    pbar.close()

    arr = np.array(scores)
    arr = arr.reshape([-1, 1])
    return arr
