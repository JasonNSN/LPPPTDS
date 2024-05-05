import numpy as np
import cv2
import matplotlib.pyplot as plt


def hist_match_single_channel(origin, template):
    """直方图匹配"""
    origin_shape = origin.shape  # 保存图像原始维数
    # 扁平化
    origin = origin.ravel()
    template = template.ravel()
    # 灰度级统计
    origin_val, new_origin_idx, origin_val_count = np.unique(origin, return_inverse=True, return_counts=True)
    template_val, template_val_count = np.unique(template, return_counts=True)
    # 计算累积直方图
    origin_cumulative_hist = np.cumsum(origin_val_count).astype(np.float64)
    origin_cumulative_hist /= origin_cumulative_hist[-1]
    template_cumulative_hist = np.cumsum(template_val_count).astype(np.float64)
    template_cumulative_hist /= template_cumulative_hist[-1]
    # 一维线性插值实现直方图匹配
    res = np.interp(origin_cumulative_hist, template_cumulative_hist, template_val)
    return res[new_origin_idx].reshape(origin_shape)


def histogram_match(origin, template, output):
    for c in range(3):
        output[:, :, c] = hist_match_single_channel(origin[:, :, c], template[:, :, c])  # 三通道各自直方图匹配
    return output


def draw_hist(img0, img1, img2):
    """绘制直方图"""
    plt.subplot(1, 3, 1)
    hist = cv2.calcHist([img0], [0], None, [256], [0, 255])  # 蓝色通道（由第2个参数控制）
    plt.plot(hist)
    plt.title('source')
    plt.subplot(1, 3, 2)
    hist = cv2.calcHist([img1], [0], None, [256], [0, 255])  # 蓝色通道（由第2个参数控制）
    plt.plot(hist)
    plt.title('template')
    plt.subplot(1, 3, 3)
    hist = cv2.calcHist([img2], [0], None, [256], [0, 255])  # 蓝色通道（由第2个参数控制）
    plt.plot(hist)
    plt.title('result')
    plt.show()


if __name__ == '__main__':
    src = cv2.imread('./data_src/origin.jpg')
    template = cv2.imread('./data_src/template.jpg')
    res = cv2.imread('./data_src/res.jpg')
    draw_hist(src, template, res)
