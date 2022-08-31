import torch
import numpy as np
import copy

from utils import s_space_code_process


def get_numpy(tensor):
    """
    将tensor转换为numpy数组
    :param tensor:
    :return:
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError(f'Unsupported input type `{type(tensor)}`!')


def sample_codes(generator, num, latent_space_type='Z', seed=0):
    """Samples latent codes randomly."""
    np.random.seed(seed)
    # 生成Z空间的codes，返回的是一个一个元素的列表
    codes = generator.sample(num)
    if latent_space_type == 'Z':
        return codes
    # 转换到GPU，之后转换到其他空间
    if latent_space_type == 'W':
        codes = transfer_latent_space(generator, codes, "Z", "W")
    elif latent_space_type.startswith('S'):
        codes = transfer_latent_space(generator, codes, "Z", "S")
    return codes


def transfer_latent_space(generator, codes, origin_space, dest_space):
    with torch.no_grad():
        if origin_space == "Z":
            # codes 为 list of [batch， dim]的array
            # 转换为tensor
            codes = [torch.from_numpy(code).type(torch.FloatTensor).cuda() for code in codes]
            # 转换到W空间
            codes = generator.mapping(codes)
            if dest_space == "W":
                codes = [get_numpy(code) for code in codes]
            elif dest_space == "S":
                codes = generator.truncate(codes, truncation=generator.config["truncation"])
                s_codes, torgb_codes = generator.s_space_encoder(codes)
                s_codes = [get_numpy(code) for code in s_codes]
                torgb_codes = [get_numpy(code) for code in torgb_codes]
                codes = [s_codes, torgb_codes]
            else:
                raise SystemExit(f"No such space {dest_space}")
        elif origin_space == "W":
            if dest_space == "S":
                codes = [torch.from_numpy(code).type(torch.FloatTensor).cuda() for code in codes]
                codes = generator.truncate(codes, truncation=generator.config["truncation"])
                s_codes, torgb_codes = generator.s_space_encoder(codes)
                s_codes = [get_numpy(code) for code in s_codes]
                torgb_codes = [get_numpy(code) for code in torgb_codes]
                codes = [s_codes, torgb_codes]
            else:
                raise SystemExit(f"No such space {dest_space}")
        else:
            raise SystemExit(f"No such space {origin_space}")
    return codes


# 将原本的codes加上boundaries*eval的变化
def get_new_codes(codes, boundaries, eval, latent_space="Z", is_tensor=False):
    if is_tensor is False:
        new_codes = copy.deepcopy(codes)

    # Z W空间的输入比较特殊，是一个list，需要单独深拷贝
    if latent_space == "Z" or latent_space == "W":
        if is_tensor is True:
            new_codes[0] = codes[0].clone()
        new_codes[0] += boundaries * eval
        return new_codes

    if is_tensor is True:
        new_style_codes = []
        new_torgb = []
        for code in codes[0]:
            new_style_codes.append(code.clone())
        for code in codes[1]:
            new_torgb.append(code.clone())
        new_codes = [new_style_codes, new_torgb]

    if latent_space == "S_ST":
        if is_tensor is False:
            new_codes = s_space_code_process.cat_style_trgb(new_codes[0], new_codes[1])
        else:
            new_codes = s_space_code_process.tensor_cat_style_trgb(new_codes[0], new_codes[1])
        new_codes += boundaries * eval
        new_codes = s_space_code_process.parse_codes(new_codes, 0)
    elif latent_space == "S_S":
        if is_tensor is False:
            new_style_codes = s_space_code_process.cat_codes(new_codes[0])
        else:
            new_style_codes = s_space_code_process.tensor_cat(new_codes[0])
        new_style_codes += boundaries * eval
        new_style_codes = s_space_code_process.parse_codes(new_style_codes, 1)
        new_codes = [new_style_codes[0], new_codes[1]]
    elif latent_space == "S_T":
        if is_tensor is False:
            new_style_codes = s_space_code_process.cat_codes(new_codes[1])
        else:
            new_style_codes = s_space_code_process.tensor_cat(new_codes[1])
        new_style_codes += boundaries * eval
        new_style_codes = s_space_code_process.parse_codes(new_style_codes, 2)
        new_codes = [new_codes[0], new_style_codes[1]]
    return new_codes
