from segmentation import mouth_detect
from utils.batch_process import get_file_path
from tqdm import *
import numpy as np
import cv2
import os


def gen_mask(image_dir, batch_num, use_ellipse=False):
    mask_dir = os.path.join(image_dir, "masks")
    if use_ellipse:
        mask_dir += "_ellipse"
    else:
        mask_dir += "_border"
    os.mkdir(mask_dir)
    detector = mouth_detect.MouseDetector()
    batch_bar = tqdm(total=batch_num)
    for i in range(batch_num):
        paths = get_file_path(image_dir, i)
        for path in paths:
            basename = os.path.basename(path)
            mask = detector.segment(path, use_ellipse)
            cv2.imwrite(f"{mask_dir}/{basename}", mask)
        batch_bar.update(1)


def gen_mask_with_hull(image_dir, batch_num, use_ellipse=False):
    mask_dir = os.path.join(image_dir, "masks_with_hull")
    if use_ellipse:
        mask_dir += "_ellipse"
    else:
        mask_dir += "_border"
    os.mkdir(mask_dir)
    detector = mouth_detect.MouseDetector()
    batch_bar = tqdm(total=batch_num)
    for i in range(batch_num):
        paths = get_file_path(image_dir, i)
        for path in paths:
            basename = os.path.basename(path)
            mask = detector.mask_hull_and_border(path, use_ellipse)
            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(f"{mask_dir}/{basename}", mask)
        batch_bar.update(1)


if __name__ == "__main__":
    import numpy as np
    d = mouth_detect.MouseDetector()
    # img = d.detect("sample/07_27_06_17/000001.png", rescale=1)
    # img = d.segment("sample/07_27_06_17/000001.png")
    img = d.mask_hull_and_border("D:\A-GAN\StyleIntervention\sample\\08_25_10_09\\013_002.png", True)
    img = cv2.resize(img, (512, 512))
    img = (img*255).astype(np.uint8)
    cv2.imwrite("D:\A-GAN\StyleIntervention\sample\\08_25_10_09\\013_002_mask_err.png", img)
    # img = cv2.imread("D:\A-GAN\StyleIntervention\sample\08_25_10_09\000000.png")
    cv2.imshow("Image", img)
    cv2.waitKey(0)  # 任意键退出
    cv2.destroyAllWindows()  # 销毁窗口