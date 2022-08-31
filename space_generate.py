#  用于指定隐空间进行实验
import argparse
import datetime
import os
import numpy as np
import cv2
import torch
from tqdm import *
from models import stylegan2_generator
from utils import s_space_code_process
from utils.logger import get_temp_logger
from utils.codes_manipulation import get_numpy


def synthesis(output_dir, img_num=1000, batch_size=50, input_codes=None, latent_space="Z", random_sample=True):
    """
    根据指定空间的codes生成图片并保存，codes也进行保存
    如果input_codes不是None，则直接使用这个作为网络的输入
    如果为None的话，则考虑random_sample，查看如何生成输入的codes
        如果random—sample设为false，则所有的codes都由Z空间产生，变换到相应的空间
        设为True（默认）的话，则交给生成器随机生成该空间的codes
    :param output_dir:
    :param img_num:
    :param batch_size:
    :param input_codes: 输入
    :param latent_space: 进行实验的隐空间
    :param random_sample: input_codes=None时，是否随机生成指定空间的codes
    :return:
    """
    logger = get_temp_logger(logger_name='space_generator')

    # 创建输出文件夹
    os.mkdir(output_dir)
    # 创建一个存储latent code的子文件夹
    codes_path = os.path.join(output_dir, "codes")
    os.mkdir(codes_path)

    if input_codes is not None:
        if latent_space == "Z" or latent_space == "W":
            img_num = input_codes[0].shape[0]
        elif latent_space == "WP":
            img_num = input_codes.shape[0]
        elif latent_space == "S":
            img_num = input_codes[0][0].shape[0]

    batch_num = img_num // batch_size

    logger.info(f"开始生成图片，一共有{batch_num}个批次, 从{latent_space}空间开始")
    g = stylegan2_generator.StyleGAN2Generator()

    is_codes_sampled = False
    if input_codes is None:
        is_codes_sampled = True
        if random_sample:
            input_codes = g.sample(img_num, latent_space=latent_space)
        else:
            # 从Z空间生成的codes转换到指定空间
            input_codes = g.sample(img_num, latent_space="Z")
    logger.info(f"成功生成随机样本，样本的类型是{type(input_codes)}")

    for i in trange(batch_num):
        if is_codes_sampled is True and random_sample is False:
            print("Sample from Z and transfer to dest space")
        inputc = None
        if latent_space == "Z" or latent_space == "W" or (is_codes_sampled is True and random_sample is False):
            inputc = [input_codes[0][i * batch_size:(i + 1) * batch_size]]
        elif latent_space == "WP":
            inputc = input_codes[i * batch_size:(i + 1) * batch_size]
        elif latent_space == "S":
            s = input_codes[0]
            trgb = input_codes[1]
            s = [code[i * batch_size:(i + 1) * batch_size] for code in s]
            trgb = [code[i * batch_size:(i + 1) * batch_size] for code in trgb]
            inputc = [s, trgb]

        logger.info(f"第{i}个批次开始生成图片")
        with torch.no_grad():
            if is_codes_sampled is True and random_sample is False:
                images, space_codes = g.synthesis(inputc, latent_space="Z")
            else:
                images, space_codes = g.synthesis(inputc, latent_space=latent_space)
        logger.info(f"图片生成完毕")

        # 保存图片
        logger.info("开始保存图片")
        images = stylegan2_generator.postprocess_image(images)
        for j in range(images.shape[0]):
            # :03d 表示一共有三位，用零填充，默认右对齐，相当于 :0>3d
            cv2.imwrite(f"{output_dir}/{i:03d}_{j:03d}.png", images[j])
        logger.info("图片保存完毕")

        logger.info("开始保存latent code")
        if latent_space == "Z":
            codes = space_codes["Z"]
            if len(codes) > 1:
                raise ValueError("本函数中不考虑style mixing, 该情况还未处理")
            codes = get_numpy(codes[0])
            np.save(f"{codes_path}/{i:03d}", codes)
        elif latent_space == "W":
            codes = space_codes["W"]
            if len(codes) > 1:
                raise ValueError("本函数中不考虑style mixing, 该情况还未处理")
            codes = get_numpy(codes[0])
            np.save(f"{codes_path}/{i:03d}", codes)
        elif latent_space == "WP":
            codes = space_codes["WP"]
            codes = get_numpy(codes)
            np.save(f"{codes_path}/{i:03d}", codes)
        elif latent_space == "S":
            s_codes = space_codes["S"]
            torgb_codes = space_codes["S_TRGB"]
            s_codes = [get_numpy(code) for code in s_codes]
            torgb_codes = [get_numpy(code) for code in torgb_codes]

            space_trgb_codes = s_space_code_process.cat_style_trgb(s_codes, torgb_codes)
            s_codes = s_space_code_process.cat_codes(s_codes)
            torgb_codes = s_space_code_process.cat_codes(torgb_codes)

            np.save(f"{codes_path}/{i:03d}_s_t", space_trgb_codes)
            np.save(f"{codes_path}/{i:03d}_s", s_codes)
            np.save(f"{codes_path}/{i:03d}_t", torgb_codes)
        else:
            raise SystemExit(f"隐空间`{latent_space}`不合法")
        logger.info("latent code保存完毕")


def easy_synthesis(codes, latent_space="Z"):
    g = stylegan2_generator.StyleGAN2Generator()
    images, _ = g.synthesis(codes, latent_space=latent_space)
    images = stylegan2_generator.postprocess_image(images)
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据指定隐空间生成随机图片")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='保存结果的路径')

    parser.add_argument("-b", "--batch_size", type=int, default=50,
                        help="每一批生成图片的数量", )

    parser.add_argument("-n", "--img_num", type=int, default=1000,
                        help="生成图片的总数量")

    parser.add_argument("--latent_space", type=str, default="Z",
                        help="要进行实验的隐空间，最终只会保存这个隐空间之后的数据")

    # 未输入--random_sample则为False，输入参数则为True
    parser.add_argument("--random_sample", action="store_true",
                        help="是否直接从指定空间随机生成codes，为False则在Z空间生成代码后通过生成器转换到指定空间")

    args = parser.parse_args()

    args.output_dir = args.output_dir + datetime.datetime.now().strftime('_%m_%d_%H_%M')

    synthesis(output_dir=args.output_dir, img_num=args.img_num, batch_size=args.batch_size,
              latent_space=args.latent_space, random_sample=args.random_sample)
