from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
import torch
import re
import moviepy.editor as mp
from predict import predict


def im_preprocess(im):
    """归一化图像数据至[0, 1]"""
    if im.max() > 1.:
        # uint8转float
        im = np.divide(im, 255., dtype=np.float16)
    assert im.max() <= 1.
    return im


def extract_audio(video_path, des_audio_dir):
    my_clip = mp.VideoFileClip(video_path)
    audio_path = des_audio_dir + '/' + re.split('/|\.', video_path)[-2] + '.mp3'
    my_clip.audio.write_audiofile(audio_path)
    print("音频已保存至：" + audio_path)
    return audio_path


class ThreadImageTamperingDetection(QThread):
    # 自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    processSignal = pyqtSignal(int)

    def __init__(self, src_video, des_video_dir, model_yolo, model_cnn, parent=None):
        super(ThreadImageTamperingDetection, self).__init__(parent)
        self.src_video = src_video
        self.des_video_dir = des_video_dir
        self.model_yolo = model_yolo
        self.model_cnn = model_cnn
        self.frame_count = 0

    def detect(self, file, model_yolo, model_cnn, des_dir):
        """
        篡改检测，先检测车牌，后判断是否篡改
        :param file: 视频路径
        :param model_yolo: YOLO目标检测模型文件
        :param model_cnn: CNN篡改检测（分类）模型文件
        :param des_dir: 要写入的目标文件夹
        """
        cap = cv2.VideoCapture(file)  # 读取视频
        if cap.isOpened():
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_name = file.split('/')[-1]
            des_video_name = video_name.split('.')[0] + '_without_audio.' + video_name.split('.')[1]
            writer = cv2.VideoWriter(des_dir + '/{}'.format(des_video_name), fourcc, fps,
                                     (int(frame_width), int(frame_height)), True)
            idx = 0
            while True:
                ret, frame = cap.read()  # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
                if not ret:  # 若是读取失败
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bboxes, _, _ = predict(frame, model_yolo)  # ltx, lty, rbx, rby
                for i in range(bboxes.shape[0]):
                    cur_bbox = bboxes[i, :]
                    ltx, lty, rbx, rby = int(cur_bbox[0]), int(cur_bbox[1]), int(cur_bbox[2]), int(cur_bbox[3])
                    plate_cut = frame[lty:rby, ltx:rbx]
                    if torch.cuda.is_available():
                        model_cnn = model_cnn.cuda()  # 在GPU上跑代码
                    testdata = []
                    dim = (100, 100)
                    img = cv2.resize(plate_cut, dim)
                    img = im_preprocess(img)
                    testdata.append(img)
                    testdata = np.array(testdata, dtype=np.float32)
                    testdata = np.transpose(testdata, (0, 3, 1, 2))
                    testdata = torch.from_numpy(testdata)
                    if torch.cuda.is_available():
                        testdata = testdata.cuda()
                    with torch.no_grad():
                        out = model_cnn(testdata)
                        out = out.squeeze(-1)
                    pred = torch.where(out > 0.5, 1, 0)  # 以0.5作为阈值
                    result = pred.to('cpu').numpy()[0]
                    if result == 1:
                        # 被篡改的车牌，高光显示
                        blk = np.zeros(frame.shape, np.uint8)
                        cv2.rectangle(blk, (ltx, lty), (rbx, rby), (255, 255, 0), -1)
                        frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)  # 读取图片后一帧帧写入到视频中
                self.processSignal.emit(idx)
                idx += 1
            writer.release()

    def thread_detection(self):
        print("开始逐帧检测被篡改车牌")
        self.detect(self.src_video, self.model_yolo, self.model_cnn, self.des_video_dir)
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
        self.thread_detection()
