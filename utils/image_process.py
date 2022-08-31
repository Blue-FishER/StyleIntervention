import numpy as np
import io
import IPython.display
import cv2
import PIL.Image

from models import stylegan2_generator


def build_generator():
    return stylegan2_generator.StyleGAN2Generator()


def easy_synthesis(codes, generator=None, latent_space="Z"):
    """返回RGB通道的图片"""
    if generator is None:
        generator = build_generator()
        # S空间可能有S_ST S_S S_T三种
    if latent_space.startswith("S"):
        latent_space = "S"
    images, _ = generator.synthesis(codes, latent_space=latent_space)
    images = stylegan2_generator.postprocess_image(images)
    return images[..., ::-1]


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)
    show_full_image(fused_image)
    return fused_image


def show_full_image(image):
    # 输入为[h，w，c]大小的numpy数组
    data = io.BytesIO()
    PIL.Image.fromarray(image).save(data, 'jpeg')
    im_data = data.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))


def save_img(images, file_path):
    # 以BGR形式存储
    cv2.imwrite(file_path, images[..., ::-1])
