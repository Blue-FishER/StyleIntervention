import cv2
import dlib
from imutils import face_utils
import os
import numpy as np
import time
import datetime


class MouseDetector():
    def __init__(self, landmark_level=0):
        # 向外扩展5像素
        self.border = 5
        # 嘴巴的起点下标、终点下标
        self.MOUSE_START = 49 - 1
        self.MOUSE_END = 68 - 1

        self.landmark_level = landmark_level
        if landmark_level == 0:
            self.shape_detector_path = os.path.join("segmentation/dat", 'shape_predictor_68_face_landmarks.dat')
        else:
            raise SystemExit(f'关键点只有0和1两种类型，目前1类型还未加入')

        # face detector
        self.detector = dlib.get_frontal_face_detector()
        # landmark detector
        self.predictor = dlib.shape_predictor(self.shape_detector_path)

    def detect(self, img_path, border=None, rescale=1, save=False):
        """
        检测指定图片的人脸中的嘴巴并使用线条标注

        :param save:
        :param img_path:
        :param border:
        :param rescale:
        :return:
        """
        border = border if border is not None else self.border

        img = cv2.imread(img_path)
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        time_start = time.time()
        faces = self.detector(gray, self.landmark_level)
        for face in faces:
            landmarks = self.predictor(gray, face)  # 关键点预测值
            # 转为numpy数组
            points = face_utils.shape_to_np(landmarks)
            mouse_points = points[self.MOUSE_START:self.MOUSE_END + 1]
            # 得到凸点范围，即为嘴巴外围左边点
            mouseHull = cv2.convexHull(mouse_points)
            # 绘制多边形轮廓，嘴巴范围
            cv2.drawContours(img, [mouseHull], -1, (0, 255, 0), 1)  # 绘制多边形轮廓，嘴巴范围
            # 近似替代为矩形
            xr, yr, wr, hr = cv2.boundingRect(mouseHull)
            # 绘制矩形，嘴巴范围
            cv2.rectangle(img, (xr - border, yr - border), (xr + wr + border, yr + hr + border), (0, 255, 9), 2)

        time_end = time.time()
        print("It takes {} to detect with dlib model!".format(time_end - time_start))

        height, width, channels = img.shape
        img = cv2.resize(img, (int(width * rescale), int(height * rescale)))

        # 保存图片
        if save:
            file_name = os.path.basename(img_path)
            file_name = datetime.datetime.now().strftime('%m_%d_%H_%M_') + file_name
            save_path = os.path.join("segmentation/images", file_name)
            cv2.imwrite(save_path, img)
        return img

    def segment(self, img_path, save=False, use_ellipse=False):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        mask = np.zeros([height, width])

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        time_start = time.time()
        faces = self.detector(gray, self.landmark_level)
        for face in faces:
            landmarks = self.predictor(gray, face)  # 关键点预测值
            # 转为numpy数组
            points = face_utils.shape_to_np(landmarks)
            mouse_points = points[self.MOUSE_START:self.MOUSE_END + 1]
            # 得到凸包，即为嘴巴外围点
            mouseHull = cv2.convexHull(mouse_points)
            # 绘制多边形轮廓，嘴巴范围
            if use_ellipse:
                ellipse = cv2.fitEllipse(mouseHull)
                center = ellipse[0]
                b, a = ellipse[1]
                angle = ellipse[2]
                a *= 1.3
                b *= 2.2
                cv2.ellipse(mask, [center, (b, a), angle], 255, cv2.FILLED)
            else:
                # 近似替代为矩形
                xr, yr, wr, hr = cv2.boundingRect(mouseHull)
                # 绘制矩形，嘴巴范围
                cv2.rectangle(mask, (xr - self.border, yr - self.border), (xr + wr + self.border, yr + hr + self.border),
                              255, cv2.FILLED)

        time_end = time.time()
        print("It takes {} to detect with dlib model!".format(time_end - time_start))

        # 保存图片
        if save is True:
            file_name = os.path.basename(img_path)
            file_name = datetime.datetime.now().strftime('%m_%d_%H_%M_') + "mask_" + file_name
            save_path = os.path.join("segmentation/images", file_name)
            cv2.imwrite(save_path, mask)
        return mask

    def mask_hull_and_border(self, img_path, use_ellipse=False):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        mask = np.zeros([height, width])

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, self.landmark_level)
        for face in faces:
            landmarks = self.predictor(gray, face)  # 关键点预测值
            # 转为numpy数组
            points = face_utils.shape_to_np(landmarks)
            mouse_points = points[self.MOUSE_START:self.MOUSE_END + 1]
            # 得到凸包，即为嘴巴外围点
            mouseHull = cv2.convexHull(mouse_points)

            if use_ellipse is False:
                # 先绘制最嘴部周边区域，近似替代为矩形
                xr, yr, wr, hr = cv2.boundingRect(mouseHull)
                # 绘制矩形，嘴巴范围
                cv2.rectangle(mask, (xr - self.border, yr - self.border), (xr + wr + self.border, yr + hr + self.border),
                              0.5, cv2.FILLED)
            else:
                ellipse = cv2.fitEllipse(mouseHull)
                center = ellipse[0]
                b,a = ellipse[1]
                angle = ellipse[2]
                a*=1.3
                b*=2.2
                cv2.ellipse(mask, [center, (b,a), angle], 0.5, cv2.FILLED)
            # 绘制真正的嘴巴范围
            cv2.drawContours(mask, [mouseHull], -1, 1, cv2.FILLED)

        return mask
