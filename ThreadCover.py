from predict import predict
from PyQt5.QtCore import QThread, pyqtSignal
from histogram_match import histogram_match
import torch
import vlpr
import re
import moviepy.editor as mp
from PlateGenerator import *
from scipy.optimize import linear_sum_assignment


def compute_IoU_matrix(bbox1, bbox2):
    """
    计算两个bbox数组的IoU矩阵（二分图）
    :param bbox1: [[lt_x2, lt_y2, rb_x2, rb_y2]]，当前帧的bbox数组
    :param bbox2: [[lt_x1, lt_y1, rb_x1, rb_y1]]，前一帧的bbox数组
    :return: 二分图bbox的交并比矩阵（i为当前帧的目标，j为前一帧的目标）
    """
    res = []
    for i in range(bbox1.shape[0]):
        temp_res = []
        for j in range(bbox2.shape[0]):
            x1_min, y1_min, x1_max, y1_max = bbox1[i, 0], bbox1[i, 1], bbox1[i, 2], bbox1[i, 3]
            x2_min, y2_min, x2_max, y2_max = bbox2[j, 0], bbox2[j, 1], bbox2[j, 2], bbox2[j, 3]
            # 分别计算两个bbox的面积
            s1 = (y1_max - y1_min + 1.) * (x1_max - x1_min + 1.)
            s2 = (y2_max - y2_min + 1.) * (x2_max - x2_min + 1.)
            # 计算相交部分的坐标
            x_min = max(x1_min, x2_min)
            y_min = max(y1_min, y2_min)
            x_max = min(x1_max, x2_max)
            y_max = min(y1_max, y2_max)
            # 计算相交部分的宽度和高度
            inter_w = max(x_max - x_min + 1, 0)
            inter_h = max(y_max - y_min + 1, 0)
            # 计算相交部分的面积
            intersection = inter_h * inter_w
            # 计算相并部分的面积
            union = s1 + s2 - intersection
            # 计算交并比
            IoU = intersection / union
            temp_res.append(IoU)
        res.append(temp_res)
    return np.array(res)


def get_best_match(matrix, IoU_threshold):
    """
    通过KM算法，求出二分图的最佳匹配（最大IoU）
    :param matrix: 记录前一帧某bbox-当前帧某bbox的IoU
    :IoU_threshold: Iou阈值
    :return: 匹配后的下标元组列表
    """
    row_idx, col_idx = linear_sum_assignment(matrix, True)
    res_list = []
    for i in range(row_idx.shape[0]):
        if matrix[row_idx[i], col_idx[i]] >= IoU_threshold:
            res_list.append((row_idx[i], col_idx[i]))
    return res_list


def extract_audio(video_path, des_audio_dir):
    my_clip = mp.VideoFileClip(video_path)
    audio_path = des_audio_dir + '/' + re.split('/|\.', video_path)[-2] + '.mp3'
    my_clip.audio.write_audiofile(audio_path)
    print("音频已保存至：" + audio_path)
    return audio_path


class ThreadCover(QThread):
    # 自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    processSignal = pyqtSignal(int)

    def __init__(self, src_video, des_video_dir, model_yolo, model_crnn, IoU_threshold, parent=None):
        super(ThreadCover, self).__init__(parent)
        self.src_video = src_video
        self.des_video_dir = des_video_dir
        self.model_yolo = model_yolo
        self.model_crnn = model_crnn
        self.IoU_threshold = IoU_threshold
        self.frame_count = 0

    def cover(self, src_video, des_video_dir, model_yolo, model_crnn, IoU_threshold):
        """
        生成车牌遮盖
        1. 利用匈牙利算法和上一帧各目标做匹配（借助上一帧车牌-bbox坐标字典）
        2. 识别出车牌的四个角点
        3. 透视变换，矫正车牌，以便做车牌识别和直方图匹配
        4. 对于匹配到的车牌（且合规，防止前一帧不完整的车牌影响后一帧完整的车牌），根据上一帧车牌转6，对于匹配不到的车牌，转5
        5. 车牌号识别，得出车牌号、车牌颜色信息（可根据车牌号长度辅助）和省份信息
        6. 若缓存中有该车牌信息，则找出对应的虚假车牌号，生成虚假车牌
        7. 缓存中没有的车牌，转8
        8. 根据颜色和省份生成虚假车牌
        9. 保存该车牌-伪造车牌信息至缓存，更新上一帧车牌字典
        src_video: 源视频文件路径
        des_video_dir：要存放处理好的视频的文件夹
        model_yolo: YOLO模型文件
        model_crnn: CRNN模型文件
        IoU_threshold: IoU阈值
        """
        cap = cv2.VideoCapture(src_video)  # 读取视频
        frame_count = 0  # 视频帧数初始化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        plates_dict = {}  # 缓存车牌信息
        pre_plates_dict = {}  # 存储上一帧车牌-bbox键值对
        if cap.isOpened():
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 视频高度
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 视频宽度
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频帧数
            self.frame_count = frame_count
            fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_name = src_video.split('/')[-1]
            des_video_name = video_name.split('.')[0] + '_without_audio.' + video_name.split('.')[1]
            writer = cv2.VideoWriter(des_video_dir + '/{}'.format(des_video_name), fourcc, fps,
                                     (int(frame_width), int(frame_height)), True)
            idx = 0  # 当前帧索引（第idx帧）
            for _ in range(int(frame_count)):
                ret, frame = cap.read()  # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
                plates = []  # 存放当前帧的所有车牌
                if not ret:  # 若是读取失败
                    break
                print("第{}帧，共{}帧".format(str(idx + 1), str(int(frame_count))))
                bboxes, cls, points = predict(frame, model_yolo)  # bbox为二维数组
                count = points.shape[0]  # 当前帧车牌数
                # 匈牙利算法做匹配
                if idx > 0:
                    pre_bboxes = []
                    for plate, bbox in pre_plates_dict.items():
                        pre_bboxes.append(bbox)
                    pre_bboxes = np.array(pre_bboxes)
                    IoU_matrix = compute_IoU_matrix(bboxes, pre_bboxes)
                    # 计算出两帧之间的最优匹配
                    if IoU_matrix.size == 0:
                        writer.write(frame)
                        idx += 1
                        self.processSignal.emit(idx)
                        continue
                    best_match = get_best_match(IoU_matrix, IoU_threshold)
                    cur_matched = []  # 当前帧匹配上的车牌
                    pre_matched = []  # 前一帧匹配上的车牌
                    for cur, pre in best_match:
                        cur_matched.append(cur)
                        pre_matched.append(pre)
                if count > 0 and points.shape[1] > 0:
                    result = frame.copy()
                    for i in range(count):
                        point_arr_cur = points[i, :, :]  # 当前车牌的角点
                        # 指定车牌的大小
                        lp_width = 880
                        lp_height = 280
                        mask_white = np.full((280, 880, 3), 255, dtype=np.uint8)  # 用于构建与车牌进行透视变换后相同位置的掩码
                        # 创建车牌上的四个点（左上角，右上角，右下角，左下角）
                        lp_points = np.array([[0, 0], [lp_width, 0], [lp_width, lp_height], [0, lp_height]],
                                             dtype=np.float32)
                        matrix_hist = cv2.getPerspectiveTransform(point_arr_cur, lp_points)  # 用于直方图匹配模板图像的透视变换矩阵
                        template = cv2.warpPerspective(frame, matrix_hist, (lp_width, lp_height))  # 原始车牌经透视变换矫正后的图像
                        plate_num = ''  # 车牌号初始化
                        if idx > 0 and i in cur_matched:
                            # 匹配到上一帧的车牌
                            plate_num = list(pre_plates_dict)[pre_matched[cur_matched.index(i)]]
                            if len(plate_num) < 7:
                                # 匹配到不合规车牌，重新识别
                                plate_num = vlpr.predict(device, model_crnn, template)
                        else:
                            # 车牌号识别
                            plate_num = vlpr.predict(device, model_crnn, template)
                        plates.append(plate_num)
                        plate_color = 0
                        is_cover = 0  # 是否遮盖（不替换）
                        if len(plate_num) == 7 and (provinces.get(plate_num[0]) is not None):
                            plate_color = 1  # 蓝牌
                            lp_width = 880
                            lp_height = 280
                            # 查找车牌号-虚假车牌号缓存
                            if plates_dict.get(plate_num) is None:
                                # 加入缓存
                                plate_num_fake = generate_plate_num(plate_color, plate_num[0])
                                plates_dict[plate_num] = plate_num_fake
                            else:
                                # 直接查找
                                plate_num_fake = plates_dict.get(plate_num)
                            lp_img_origin = generate_plate(plate_num_fake, '')
                        elif len(plate_num) == 8 and (provinces.get(plate_num[0]) is not None):
                            plate_color = 0  # 绿牌
                            lp_width = 960
                            lp_height = 280
                            # 查找车牌号-虚假车牌号缓存
                            if plates_dict.get(plate_num) is None:
                                # 加入缓存
                                plate_num_fake = generate_plate_num(plate_color, plate_num[0])
                                plates_dict[plate_num] = plate_num_fake
                            else:
                                # 直接查找
                                plate_num_fake = plates_dict.get(plate_num)
                            lp_img_origin = generate_plate(plate_num_fake, '')
                        else:
                            # 不替换，转遮盖
                            # 通过车牌颜色取样获得车牌颜色
                            is_cover = 1
                            height, width = template.shape[0], template.shape[1]
                            # 取18个采样点（3行6列）
                            x_sample = np.arange(0, width, width // 6, int)[1:]
                            y_sample = np.arange(0, height, height // 3, int)[1:]
                            loc = np.array(np.meshgrid(y_sample, x_sample)).T.reshape(-1, 2)  # 获取笛卡尔积
                            b_arr = template[loc[:, 0], loc[:, 1], 0]  # 蓝色分量
                            g_arr = template[loc[:, 0], loc[:, 1], 1]  # 绿色分量
                            judge_arr = np.array(b_arr > g_arr)
                            unique, count = np.unique(judge_arr, return_counts=True)
                            data_count = dict(zip(unique, count))
                            count_b = data_count.get(True)
                            count_g = data_count.get(False)
                            if count_b is None:
                                count_b = 0
                            if count_g is None:
                                count_g = 0
                            plate_type = 1 if count_b > count_g else 0  # 1蓝0绿
                            lp_img = np.full((280, 880, 3), (155, 60, 41), dtype=np.uint8) \
                                if plate_type == 1 else np.full((280, 960, 3), (171, 251, 102), dtype=np.uint8)
                        if is_cover == 0:
                            # 车牌直方图匹配
                            origin = np.copy(lp_img_origin)  # 原始虚假车牌图像
                            for c in range(3):
                                origin[:, :, c] = cv2.equalizeHist(origin[:, :, c])  # 直方图均衡
                            output = np.copy(origin)
                            lp_img = histogram_match(origin, template, output)  # 根据原始车牌进行直方图匹配
                        frame = result
                        # 计算透视变换矩阵
                        lp_points = np.array([[0, 0], [lp_width, 0], [lp_width, lp_height], [0, lp_height]],
                                             dtype=np.float32)  # 长宽重新赋值
                        matrix = cv2.getPerspectiveTransform(lp_points, point_arr_cur)  # 将车牌透视变换到背景图中（计算变换矩阵）
                        # 进行透视变换
                        result = cv2.warpPerspective(lp_img, matrix, (frame.shape[1], frame.shape[0]))  # 变换后的车牌
                        mask_white = np.full((lp_height, lp_width, 3), 255, dtype=np.uint8)  # 用于构建与车牌进行透视变换后相同位置的掩码
                        mask = cv2.warpPerspective(mask_white, matrix, (frame.shape[1], frame.shape[0]))  # 变换后的车牌掩码
                        # 将结果图像合并到大图上
                        result[np.where(mask[:, :, 0] == 0)] = frame[np.where(mask[:, :, 0] == 0)]
                    writer.write(result)
                else:
                    writer.write(frame)
                # 更新前一帧的车牌字典
                pre_plates_dict = {}
                for i in range(len(plates)):
                    pre_plates_dict[plates[i]] = bboxes[i, :]
                self.processSignal.emit(idx)
                idx += 1
            writer.release()

    def thread_cover(self):
        print("开始逐帧检测并替换")
        self.cover(self.src_video, self.des_video_dir, self.model_yolo, self.model_crnn, self.IoU_threshold)
        video_name = self.src_video.split('/')[-1]
        video_path = self.des_video_dir + '/' + video_name.split('.')[0] + '_without_audio.' + video_name.split('.')[1]
        audio_path = extract_audio(self.src_video, self.des_video_dir)  # 提取音频并保存
        output_path = self.des_video_dir + '/' + video_name
        # 加载视频和音频文件
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        # 将音频与视频合并
        video_clip = video_clip.set_audio(audio_clip)
        # 保存带有声音的视频
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        # 关闭视频和音频文件
        video_clip.close()
        audio_clip.close()
        if self.frame_count > 0:
            self.processSignal.emit(int(self.frame_count))

    # run函数是子线程中的重写方法，线程启动后开始执行
    def run(self):
        # 发射自定义信号
        # 通过emit函数将参数i传递给主线程，触发自定义信号
        self.thread_cover()
