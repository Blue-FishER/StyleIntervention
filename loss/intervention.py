import torch
import torch.nn as nn
import numpy as np
from models import stylegan2_generator
from utils import codes_manipulation
from utils import s_space_code_process


class StyleIntervention(nn.Module):
    def __init__(self, latent_type="S_ST"):
        super(StyleIntervention, self).__init__()
        # coefficient = np.zeros(shape=[1, 9088])
        self.latent_type = latent_type
        # coefficient = np.random.randn(1, 9088)
        coefficient = np.ones(shape=[1, 9088]) * 0.5
        coefficient_list = s_space_code_process.parse_codes(coefficient, 0)
        if latent_type == "S_ST" or latent_type == "S_S":
            for i in range(len(coefficient_list[0])):
                setattr(self, f"style_{i}",
                        nn.Parameter(torch.from_numpy(coefficient_list[0][i]).type(torch.FloatTensor).cuda()))
                # nn.Parameter(torch.from_numpy(coefficient_list[0][i]).type(torch.FloatTensor)))
        if latent_type == "S_ST" or latent_type == "S_T":
            for i in range(len(coefficient_list[1])):
                setattr(self, f"torgb_{i}",
                        nn.Parameter(torch.from_numpy(coefficient_list[1][i]).type(torch.FloatTensor).cuda()))
                # nn.Parameter(torch.from_numpy(coefficient_list[1][i]).type(torch.FloatTensor)))

    def get_coefficient(self):
        style = []
        torgb = []
        # default type: S_ST
        if self.latent_type == "S_ST":
            style.append(self.style_0)
            torgb.append(self.torgb_0)
            for i in range(1, 9):
                style.append(getattr(self, f"style_{2 * i - 1}"))
                style.append(getattr(self, f"style_{2 * i}"))
                torgb.append(getattr(self, f"torgb_{i}"))

            coefficient = s_space_code_process.tensor_cat_style_trgb(style, torgb)

        elif self.latent_type == "S_S":
            style.append(self.style_0)
            for i in range(1, 9):
                style.append(getattr(self, f"style_{2 * i - 1}"))
                style.append(getattr(self, f"style_{2 * i}"))

            coefficient = s_space_code_process.tensor_cat(style)

        elif self.latent_type == "S_T":
            for i in range(9):
                torgb.append(getattr(self, f"torgb_{i}"))

            coefficient = s_space_code_process.tensor_cat(torgb)

        return coefficient

    def forward(self, delta_sz, delta_sn):
        """
        delta_sn [1, 9088/6048/3040]维度的tensor，
        delta_sz/sn [batch, 9088/6048/3040]维度的tensor，
        最终返回经过系数变换后的delta_s
        """
        coefficient = self.get_coefficient()
        if coefficient.shape[1] != delta_sz.shape[1] or coefficient.shape[1] != delta_sn.shape[1]:
            raise SystemExit(f"输入的维度和系数的维度不一致，应该为{coefficient.shape[1]}")

        return (1 - coefficient) * delta_sz + coefficient * delta_sn


class InterventionLoss(nn.Module):
    def __init__(self, generator=None, lambda_attr=1e-2, lambda_norm=1e-6, s_eval=1):
        super(InterventionLoss, self).__init__()
        self.latent_space = "S"
        self.s_eval = s_eval
        self.lambda_attr = lambda_attr
        self.lambda_norm = lambda_norm
        self.image_show = False  # 用于控制训练途中是否生成图片

        if generator is None:
            self.generator = stylegan2_generator.StyleGAN2Generator()
        else:
            self.generator = generator

    def pixel_loss(self, s, delta_s, mask, is_masks_with_hull=False):
        # 生成原本的图片时并不需要进行反向传播
        with torch.no_grad():
            images, _ = self.generator.synthesis(s, latent_space="S")

        new_codes = get_new_codes(s, delta_s, eval=self.s_eval, latent_space=self.latent_space, is_tensor=True)

        # with torch.no_grad():
        new_images, _ = self.generator.synthesis(new_codes, latent_space="S")
        if self.image_show:
            with torch.no_grad():
                tmp = stylegan2_generator.postprocess_image(images)
                if is_masks_with_hull:
                    tmp = np.concatenate((tmp, mask.cpu().permute(0, 2, 3, 1).numpy(),
                                          stylegan2_generator.postprocess_image(new_images)), axis=0)
                else:
                    m_n_tmp = stylegan2_generator.postprocess_image(torch.cat([mask, new_images], dim=0))
                    tmp = np.concatenate((tmp, m_n_tmp), axis=0)
                # tmp = stylegan2_generator.postprocess_image(new_images)
                imshow(tmp[..., ::-1], 10)
            # 自动停止图片显示，在训练时进行人为控制
            self.image_show = False

        delta_images = new_images - images
        if is_masks_with_hull is False:
            delta = (1 - mask) * delta_images
        else:
            delta = (mask == 0) * delta_images

        # 先求出每个通道的范数，最后加在一起
        per_channel_sum = torch.sqrt(torch.sum(delta ** 2, dim=[2, 3]))
        sum = torch.sum(per_channel_sum, dim=1)

        # 新增的损失,希望这个损失越大越好，编辑嘴部周边的区域更多
        if is_masks_with_hull:
            delta_around_mouth = (mask == 127) * delta_images
            new_per_channel_sum = torch.sqrt(torch.sum(delta_around_mouth ** 2, dim=[2, 3]))
            new_sum = torch.sum(new_per_channel_sum, dim=1)
            new_sum = -1 * new_sum
            sum = sum + new_sum

        return sum, new_codes

    def l2_norm_loss(self, coefficient):
        # return torch.sqrt(torch.sum(coefficient ** 2))
        # 避免梯度为无穷大
        return torch.sqrt(torch.sum(coefficient ** 2) + 1e-8)

    def attr_loss(self, delta_sn, delta_s):
        # delta_s为[batch，channels], delta_sn为[1, channels]
        dot = torch.sum(delta_sn * delta_s, axis=1)  # [batch, 1]
        norm_sn = torch.sqrt(torch.sum(delta_sn ** 2))  # number
        norm_s = torch.sqrt(torch.sum(delta_s ** 2, axis=1))  # [batch, 1]
        return -1.0 * torch.mean(dot / (norm_s * norm_sn))

    def forward(self, coefficient, s, delta_s, delta_sn, mask, is_masks_with_hull=False):
        """
        :param delta_s: s空间最终的位移
        :param coefficient: 9088/6048/3030
        :param s: list（参考gan模型synthesis的输入）
        :param delta_sn: 9088/6048/3040
        :param mask:
        :param is_masks_with_hull:masks中是否是二值图像，如果是二值，则用0和1表示；如果标注出了嘴部的凸包，则嘴内部、嘴周边、其余区域分别对-应1 127 255
        :return:
        """
        if delta_sn.shape[1] == 9088:
            self.latent_space = "S_ST"
        elif delta_sn.shape[1] == 6048:
            self.latent_space = "S_S"
        elif delta_sn.shape[1] == 3040:
            self.latent_space = "S_T"

        pixel_loss, new_codes = self.pixel_loss(s, delta_s, mask, is_masks_with_hull)
        pixel_loss = torch.mean(pixel_loss)
        attribute_loss = self.lambda_attr * self.attr_loss(delta_sn, delta_s)
        l2_norm_loss = self.lambda_norm * self.l2_norm_loss(coefficient)
        loss = pixel_loss + attribute_loss + l2_norm_loss

        # 单独训练pixel-loss
        # pixel_loss, new_codes = self.pixel_loss(s, delta_s, mask)
        # pixel_loss = torch.mean(pixel_loss)
        # loss = pixel_loss
        # attribute_loss = torch.Tensor([0])
        # l2_norm_loss = torch.Tensor([0])

        # 单独训练attribute-loss
        # attribute_loss = self.lambda_attr * self.attr_loss(delta_sn, delta_s)
        # loss = attribute_loss
        # pixel_loss = torch.Tensor([0])
        # new_codes = None
        # l2_norm_loss = torch.Tensor([0])

        # 单独训练l2_norm_loss
        # l2_norm_loss = self.lambda_norm * self.l2_norm_loss(coefficient)
        # loss = l2_norm_loss
        # pixel_loss = torch.Tensor([0])
        # new_codes = None
        # attribute_loss = torch.Tensor([0])

        return {"loss": [loss, pixel_loss, attribute_loss, l2_norm_loss],
                "new_codes": new_codes}
