import random
import numpy as np
from utils.logger import get_temp_logger
from models import model_settings
from models import stylegan2_generator_model
from op import conv2d_gradfix
import torch


def modulated_conv_s_space(conv, input, style):
    """
    由于style已经经过了仿射变换，所以StyledConv/ToRGB内部的ModulatedConv应该重构，
    返回输入经过modulate的卷积核后的输出

    :param conv: 原模型中的ModulatedConv层
    :param input: [b,c,h,w]
    :param style: 每层对应的s空间的style，大小为[b, in_channel]
    :return:
    """
    batch, in_channel, height, width = input.shape

    # 和ModulatedConv的forward函数基本相同
    if not conv.fused:
        # weight：[1, out_channel, in_channel, kernel_size, kernel_size]
        weight = conv.scale * conv.weight.squeeze(0)

        if conv.demodulate:
            # 相当于in-channel每个通道的卷积核都乘上了对应的s
            # 一共out-channel个卷积核，style被广播out次
            # weight被广播batch次
            w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
            # w [batch, out, in, k, k]
            dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

        input = input * style.reshape(batch, in_channel, 1, 1)

        # weight：[out_channel, in_channel, kernel_size, kernel_size]
        if conv.upsample:
            # 转置卷积
            weight = weight.transpose(0, 1)
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2
            )
            out = conv.blur(out)

        elif conv.downsample:
            input = conv.blur(input)
            out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

        else:
            out = conv2d_gradfix.conv2d(input, weight, padding=conv.padding)

        if conv.demodulate:
            out = out * dcoefs.view(batch, -1, 1, 1)

        return out

    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style
    # weight [batch, out, in, k, k]

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )
    # weight [batch*out, in, k, k]

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = conv2d_gradfix.conv_transpose2d(
            input, weight, padding=0, stride=2, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        # 输出的通道数也应该变为 batch * out_channel
        input = input.view(1, batch * in_channel, height, width)
        # weight [batch*out, in, k, k]
        out = conv2d_gradfix.conv2d(
            input, weight, padding=0, stride=2, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = conv2d_gradfix.conv2d(
            input, weight, padding=conv.padding, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    return out


def styled_conv_s_space(layer, input, style, noise):
    out = modulated_conv_s_space(layer.conv, input, style)
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    return out


def to_RGB_s_space(layer, input, style, skip=None):
    out = modulated_conv_s_space(layer.conv, input, style)
    out = out + layer.bias

    if skip is not None:
        out = out + layer.upsample(skip)

    return out


# 将cpu tensor(b c h w)转为numpy（BGR）
def postprocess_image(images):
    images = images.cpu().detach()
    # 将数据范围转换到0-1之间
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    # 将通道交换到最后一个维度
    images = images.permute(0, 2, 3, 1).numpy()
    images = (images * 255).astype(np.uint8)
    # RGB -> BGR
    return images[..., ::-1]


class StyleGAN2Generator(object):
    def __init__(self, logger=None):
        self.config = model_settings.CONFIG
        self.logger = logger if logger is not None else get_temp_logger()

        # 加载模型
        self.logger.info(f'Loading StyleGAN2 model from{self.config["model_path"]}')
        self.model = stylegan2_generator_model.Generator(
            size=self.config["size"],
            style_dim=self.config["style_dim"],
            n_mlp=self.config["n_mlp"]
        )
        check_point = torch.load(self.config["model_path"])
        self.model.load_state_dict(check_point["g_ema"])
        self.model.eval()
        self.model.cuda()
        self.logger.info("Model loaded successfully")

    def sample(self, num, latent_space="Z"):
        if latent_space == "Z":
            # list of [batch, dim]
            codes = [np.random.randn(num, self.config["style_dim"]).astype(np.float32)]
        elif latent_space == "W":
            # list of [batch, dim]
            codes = [np.random.randn(num, self.config["style_dim"]).astype(np.float32)]
        elif latent_space == "WP":
            # [batch, n_latent, dim]
            codes = np.random.randn(num, self.model.n_latent, self.config["style_dim"]).astype(np.float32)
        elif latent_space == "S":
            # [batch, in-channel]
            style_space = []
            trgb_space = []
            style_space.append(np.random.rand(num, self.model.channels[4]).astype(np.float32))
            trgb_space.append(np.random.rand(num, self.model.channels[4]).astype(np.float32))
            for i in range(3, 11):
                # 第一个conv的in-channel是上一层的维度
                style_space.append(np.random.rand(num, self.model.channels[2**(i-1)]).astype(np.float32))
                style_space.append(np.random.rand(num, self.model.channels[2**i]).astype(np.float32))
                trgb_space.append(np.random.rand(num, self.model.channels[2**i]).astype(np.float32))
            codes = [style_space, trgb_space]
        else:
            raise ValueError(f'隐空间`{latent_space}` 不合法!')

        return codes

    def mapping(self, samples):
        """
        将samples列表映射为W空间code的列表
        :param samples: 每个元素的大小为[batch, latent_dim]，组成一个列表
        :return:
        """
        styles = [self.model.style(sample) for sample in samples]
        return styles

    def truncate(self, styles, truncation=1,
                 truncation_latent=None,
                 inject_index=None):
        """
        把W空间映射到W+空间
        styles如果包含两个及以上的元素，则对前两个元素进行style_mixing

        :param styles:  list of [b, latent_dim]
        :param truncation:  截断系数
        :param truncation_latent:  截断的平均code
        :param inject_index:  style_mix的分割点
        :return:  [batch, n_latent, dim]
        """
        # 截断
        if truncation < 1:
            if truncation_latent is None:
                # 利用4096个随机向量生成阶段向量
                truncation_latent = self.model.mean_latent(self.config["mean_truncation_num"])
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t

        # styles：list of tensor
        # generate.py [sample, latent_dim]
        if len(styles) < 2:
            if styles[0].ndim < 3:  # 维度个数小于3
                # [sample, latent_dim] -> [sample, n_latent, latent_num]
                latent = styles[0].unsqueeze(1).repeat(1, self.model.n_latent, 1)
            else:
                latent = styles[0]
        # 如果styles多于两个，则只取前两个（style mixing）
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        # latent [batch, n_latent, dim]
        return latent

    def s_space_encoder(self, latent):
        """
        由WP空间映射到S空间（进行一次仿射变换，也就是全连接层），返回S空间向量的list

        :param latent: [batch, n_latent(18), latent_dim(512)]
        :return: list of 每层的向量[batch, in_channel]，一共有26层, 卷积层有17个，torgb有9个
        """
        style_space = []
        to_rgb_space = []

        # 获取每层的noise
        noise = [getattr(self.model.noises, 'noise_{}'.format(i)) for i in range(self.model.num_layers)]

        # modulation就是一个全连接层（EqualLinearLayer）
        # modulation返回值大小：[batch, channel]
        style_space.append(self.model.conv1.conv.modulation(latent[:, 0]))
        to_rgb_space.append(self.model.to_rgb1.conv.modulation(latent[:, 1]))

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.model.convs[::2], self.model.convs[1::2], noise[1::2], noise[2::2], self.model.to_rgbs
        ):
            style_space.append(conv1.conv.modulation(latent[:, i]))
            style_space.append(conv2.conv.modulation(latent[:, i + 1]))
            to_rgb_space.append(to_rgb.conv.modulation(latent[:, i + 2]))
            i += 2

        return style_space, to_rgb_space

    def s_space_decoder(self, style_space, to_rgb_space, latent, use_to_rgb=False):
        """
        从s空间生成最终的图片

        :param style_space: 17个卷积层的code
        :param to_rgb_space: 9个TORGB层的code
        :param latent: W+层的style code， [batch, n_latent(18),latent_dim(512)]
        :param use_to_rgb:  是否使用S空间中TORGB层的code
        :return: 最终的图像
        """
        # 获取每层的noise
        noise = [getattr(self.model.noises, 'noise_{}'.format(i)) for i in range(self.model.num_layers)]

        # 这个参数只是为了获取它的batch
        out = self.model.input(style_space[0])
        out = styled_conv_s_space(self.model.conv1, out, style_space[0], noise[0])
        if use_to_rgb:
            skip = to_RGB_s_space(self.model.to_rgb1, out, to_rgb_space[0])
        else:
            skip = self.model.to_rgb1(out, latent[:, 1])

        i = 1
        rgb_space_level = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.model.convs[::2], self.model.convs[1::2], noise[1::2], noise[2::2], self.model.to_rgbs
        ):
            out = styled_conv_s_space(conv1, out, style_space[i], noise=noise1)
            out = styled_conv_s_space(conv2, out, style_space[i + 1], noise=noise2)
            if use_to_rgb:
                skip = to_RGB_s_space(to_rgb, out, to_rgb_space[rgb_space_level], skip)
            else:
                skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2
            rgb_space_level += 1

        image = skip
        return image

    def synthesis(self, codes, latent_space="Z"):
        space_codes = {}
        # 默认不适用S空间中torgb层的code
        use_TORGB_space = False

        # 从Z空间开始生成
        if latent_space == "Z":
            if type(codes[0]) == np.ndarray:
                codes = [torch.from_numpy(code).type(torch.FloatTensor).cuda() for code in codes]
            # for code in codes:
            #     if not code.is_cuda:
            #         raise SystemExit("数据没有成功移动到GPU上")
            # codes大小为 n* [batch, latent_dim]
            space_codes["Z"] = codes

            styles = self.mapping(codes)
            space_codes["W"] = styles

            latent = self.truncate(styles, truncation=self.config["truncation"])
            space_codes["WP"] = latent

            style_space, to_rgb_space = self.s_space_encoder(latent)
            space_codes["S"] = style_space
            space_codes["S_TRGB"] = to_rgb_space

        elif latent_space == "W":
            # styles大小为 n*[batch, latent_dim]
            styles = codes
            if type(codes[0]) == np.ndarray:
                styles = [torch.from_numpy(code).type(torch.FloatTensor).cuda() for code in codes]

            space_codes["W"] = styles

            latent = self.truncate(styles, truncation=self.config["truncation"])
            space_codes["WP"] = latent

            style_space, to_rgb_space = self.s_space_encoder(latent)
            space_codes["S"] = style_space
            space_codes["S_TRGB"] = to_rgb_space

        elif latent_space == "WP":
            # latent: [batch, n_latent, latent_dim]
            latent = codes
            if type(codes) == np.ndarray:
                latent = torch.from_numpy(codes).type(torch.FloatTensor).cuda()

            space_codes["WP"] = latent

            style_space, to_rgb_space = self.s_space_encoder(latent)
            space_codes["S"] = style_space
            space_codes["S_TRGB"] = to_rgb_space

        elif latent_space == "S":
            # codes: list of 2, 一个是style_space(17个ndarray/tensor)，一个是to_rgbs_pace(9个ndarray/tensor)
            style_space = codes[0]
            to_rgb_space = codes[1]
            if type(codes[0][0]) == np.ndarray:
                style_space = [torch.from_numpy(s).type(torch.FloatTensor).cuda() for s in codes[0]]
                to_rgb_space = [torch.from_numpy(trgb).type(torch.FloatTensor).cuda() for trgb in codes[1]]
            # S: 17 [batch, channel]
            # ToRGb: 9 [batch, channel]
            space_codes["S"] = style_space
            space_codes["S_TRGB"] = to_rgb_space
            # 如果只给出S空间的code，必须要给出TORGB层的才可以生成图像
            use_TORGB_space = True
            latent = None
        else:
            raise SystemExit(f"No such latent space {latent_space}")

        # 使用解码器生成图片
        images = self.s_space_decoder(style_space, to_rgb_space, latent, use_TORGB_space)
        return images, space_codes
