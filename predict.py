import os
from ultralytics import YOLO
import numpy as np
import cv2


def predict(img_name, model):
    """
    图片关键点预测
    img_name：图片路径
    models：用作预测的模型
    return：
        boxes：[[ltx, lty, rbx, rby], [ltx, lty, rbx, rby]] 二维
        cls：类别（Label） [cls1, cls2] 一维
        points：关键点 三维（若干个二维数组）
    """
    results = model([img_name])
    cls = []
    bboxes = []
    points = []
    for result in results:
        boxes = np.array(result.boxes.data.cpu())  # bboxes.data构成：(LTx, LTy, RBx, RBy. 置信度, 类别)
        bboxes = np.array(boxes)[:, 0:4]  # Bounding Box：[[ltx, lty, rbx, rby], [ltx, lty, rbx, rby]] 二维
        cls = np.array(boxes)[:, 5]  # 类别（Label）：[cls1, cls2] 一维
        points = np.array(result.keypoints.xy.cpu())  # 关键点 三维（若干个二维数组）
    return bboxes, cls, points


if __name__ == '__main__':
    model = YOLO('models/yolo_5500.pt')
    img_name = './data_src/img.jpg'
    bboxes, cls, points = predict(img_name, model)  # 得到bbox的坐标、类别、关键点
    print('bboxes', bboxes)
    print('cls', cls)
    print('points', points)
    img = cv2.imread(img_name)
    for j in range(points.shape[0]):
        for i in range(points.shape[1]):
            cv2.circle(img, (int(points[j, i, 0]), int(points[j, i, 1])), 2, (0, 255, 255), 2)
    for i in range(bboxes.shape[0]):
        cv2.rectangle(img, tuple(map(int, bboxes[i, 0:2])), tuple(map(int, bboxes[i, 2:4])), (0, 255, 255), 2)
    # cv2.imwrite('res/' + img_name, img)
    # 显示图像
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
