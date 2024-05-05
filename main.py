import math
import sys
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from mainForm import Ui_Form
import cv2
import torch
import vlpr
from CNN import CNN
from ultralytics import YOLO
from ThreadCover import ThreadCover
from ThreadImageTamperingDetection import ThreadImageTamperingDetection


class MyWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 导入ui
        # 全局变量初始化
        self.play_swap = 0  # 播放状态，初始未播放
        self.play_detection = 0  # 播放状态，初始未播放
        self.file_name_swap = ''  # 读取的文件名
        self.file_name_detection = ''  # 读取的文件名
        self.video_len = 0  # 视频秒数
        self.media_player_swap = QMediaPlayer(None, QMediaPlayer.VideoSurface)  # 播放器（swap）
        self.media_player_swap.setNotifyInterval(10)
        self.media_player_swap.setVideoOutput(self.widget_video_swap)
        self.media_player_swap.durationChanged.connect(self.update_duration_swap)
        self.media_player_swap.positionChanged.connect(self.update_position_swap)
        self.capture_swap = None
        self.media_player_detection = QMediaPlayer(None, QMediaPlayer.VideoSurface)  # 播放器（detection）
        self.media_player_detection.setNotifyInterval(100)
        self.media_player_detection.setVideoOutput(self.widget_video_detection)
        self.media_player_detection.durationChanged.connect(self.update_duration_detection)
        self.media_player_detection.positionChanged.connect(self.update_position_detection)
        self.capture_detection = None
        self.model_yolo = YOLO('models/yolo_5500.pt', task='pose')
        print("YOLO目标检测模型载入成功")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_crnn = vlpr.init_model(device, 'models/crnn.pth')
        print("CRNN车牌文字识别模型载入成功")
        self.model_cnn = torch.load('models/cnn_tampering_detection.pth')
        print("CNN车牌篡改检测模型载入成功")
        self.swap_success = 0
        self.detection_success = 0

        # 信号与槽
        # 功能切换
        self.tbtn_swap.pressed.connect(self.to_swap)
        self.tbtn_detection.pressed.connect(self.to_detection)
        # 播放暂停
        self.btn_play_pause_swap.clicked.connect(self.play_or_pause_swap)
        self.btn_play_pause_detection.clicked.connect(self.play_or_pause_detection)
        # 进度条
        self.slider_video_swap.sliderMoved.connect(self.set_video_position_swap)
        self.slider_video_detection.sliderMoved.connect(self.set_video_position_detection)
        self.slider_video_swap.customSliderClicked.connect(self.slider_pressed_swap)
        self.slider_video_detection.customSliderClicked.connect(self.slider_pressed_detection)
        self.slider_video_swap.valueChanged.connect(
            lambda: self.lb_cur_time_swap.setText(
                '{:0>2d}:{:0>2d}'.format(int((self.slider_video_swap.value() // 1000) / 60),
                                         (self.slider_video_swap.value() // 1000) % 60)))
        self.slider_video_detection.valueChanged.connect(
            lambda: self.lb_cur_time_detection.setText(
                '{:0>2d}:{:0>2d}'.format(int((self.slider_video_detection.value() // 1000) / 60),
                                         (self.slider_video_detection.value() // 1000) % 60)))
        # 前进/后退若干ms按钮
        step = 100  # 前进/后退步长（ms）
        self.btn_forward_swap.clicked.connect(
            lambda: self.forward_control(self.slider_video_swap, self.media_player_swap, step,
                                         self.media_player_swap.duration()))
        self.btn_backward_swap.clicked.connect(
            lambda: self.backward_control(self.slider_video_swap, self.media_player_swap, step))
        self.btn_forward_detection.clicked.connect(
            lambda: self.forward_control(self.slider_video_detection, self.media_player_detection, step,
                                         self.media_player_detection.duration()))
        self.btn_backward_detection.clicked.connect(
            lambda: self.backward_control(self.slider_video_detection, self.media_player_detection, step))
        # 导入视频按钮
        self.btn_import.clicked.connect(self.video_import)
        # 保存视频按钮
        self.btn_save.clicked.connect(self.save_result)
        # 车牌替换按钮
        self.btn_swap.clicked.connect(self.swap)
        # 车牌篡改检测按钮
        self.btn_detection.clicked.connect(self.detection)
        # 初始不可用的按钮
        # 播放/左右按钮
        self.btn_play_pause_swap.setEnabled(False)
        self.btn_backward_swap.setEnabled(False)
        self.btn_forward_swap.setEnabled(False)
        self.slider_video_swap.setEnabled(False)
        self.btn_play_pause_detection.setEnabled(False)
        self.btn_backward_detection.setEnabled(False)
        self.btn_forward_detection.setEnabled(False)
        self.slider_video_detection.setEnabled(False)
        # 处理按钮
        self.btn_swap.setEnabled(False)
        self.btn_detection.setEnabled(False)

        self.stackedWidget.setCurrentIndex(0)  # 设置首页

    def to_swap(self):
        """切换到替换功能"""
        self.stackedWidget.setCurrentIndex(0)
        if self.play_detection == 1:
            self.play_or_pause_detection()

    def to_detection(self):
        """切换到检测功能"""
        self.stackedWidget.setCurrentIndex(1)
        if self.play_swap == 1:
            self.play_or_pause_swap()

    def play_or_pause_swap(self):
        """播放/暂停切换（替换）"""
        if self.play_swap == 0:
            self.btn_play_pause_swap.setIcon(QIcon(':/icons/resource/pause(#1ECE9E).png'))
            self.play_swap = 1  # 切换为播放状态
            self.media_player_swap.play()
        elif self.play_swap == 1:
            self.btn_play_pause_swap.setIcon(QIcon(':/icons/resource/play(#1ECE9E).png'))
            self.play_swap = 0  # 切换为暂停状态
            self.media_player_swap.pause()

    def play_or_pause_detection(self):
        """播放/暂停切换（检测）"""
        if self.play_detection == 0:
            self.btn_play_pause_detection.setIcon(QIcon(':/icons/resource/pause(#1ECE9E).png'))
            self.play_detection = 1  # 切换为播放状态
            self.media_player_detection.play()
        elif self.play_detection == 1:
            self.btn_play_pause_detection.setIcon(QIcon(':/icons/resource/play(#1ECE9E).png'))
            self.play_detection = 0  # 切换为暂停状态
            self.media_player_detection.pause()

    def handle_media_status_swap(self, status):
        """swap播放器播放完毕"""
        if status == QMediaPlayer.EndOfMedia:
            self.play_swap = 0
            self.media_player_swap.setPosition(0)
            self.media_player_swap.play()
            self.media_player_swap.pause()
            self.btn_play_pause_swap.setIcon(QIcon(':/icons/resource/play(#1ECE9E).png'))
            self.lb_cur_time_swap.setText('{:0>2d}:{:0>2d}'.format(0, 0))

    def handle_media_status_detection(self, status):
        """detection播放器播放完毕"""
        if status == QMediaPlayer.EndOfMedia:
            self.play_detection = 0
            self.media_player_detection.setPosition(0)
            self.media_player_detection.play()
            self.media_player_detection.pause()
            self.btn_play_pause_detection.setIcon(QIcon(':/icons/resource/play(#1ECE9E).png'))
            self.lb_cur_time_detection.setText('{:0>2d}:{:0>2d}'.format(0, 0))

    def video_import(self):
        """
        视频读取：
        不同功能分别读取视频
        视频读入后，滑动条最大值设为视频的ms数，实现单步100ms
        """
        if self.stackedWidget.currentIndex() == 0:
            # 替换页
            # 读取文件
            file_url = QFileDialog.getOpenFileUrl(self, caption='打开文件', filter="视频文件(*.mp4;*.avi)")[0]
            self.file_name_swap = file_url.toString()[8:]  # 文件路径
            if self.file_name_swap != '':
                # 导入时若有正在播放的视频，先暂停
                if self.play_swap == 1:
                    self.play_or_pause_swap()
                self.media_player_swap.setMedia(QMediaContent(file_url))
                self.media_player_swap.pause()
                self.capture_swap = cv2.VideoCapture(self.file_name_swap)
                rate = self.capture_swap.get(5)  # 帧速率
                frame_num = self.capture_swap.get(7)  # 视频文件的帧数
                self.frame_count_swap = frame_num
                duration = math.ceil(frame_num / rate)  # 视频总帧数/帧速率得出视频时长
                self.lb_cur_time_swap.setText('{:0>2d}:{:0>2d}'.format(0, 0))
                self.lb_max_time_swap.setText('{:0>2d}:{:0>2d}'.format(int(duration / 60), duration % 60))
                self.btn_play_pause_swap.setEnabled(True)
                self.btn_backward_swap.setEnabled(True)
                self.btn_forward_swap.setEnabled(True)
                self.slider_video_swap.setEnabled(True)
                self.btn_swap.setEnabled(True)
                self.media_player_swap.mediaStatusChanged.connect(self.handle_media_status_swap)
        elif self.stackedWidget.currentIndex() == 1:
            # 检测页
            # 读取文件
            file_url = QFileDialog.getOpenFileUrl(self, caption='打开文件', filter="视频文件(*.mp4;*.avi)")[0]
            self.file_name_detection = file_url.toString()[8:]  # 文件路径
            if self.file_name_detection != '':
                # 导入时若有正在播放的视频，先暂停
                if self.play_detection == 1:
                    self.play_or_pause_detection()
                self.media_player_detection.setMedia(QMediaContent(file_url))
                # self.media_player_detection.play()
                self.media_player_detection.pause()
                self.capture_detection = cv2.VideoCapture(self.file_name_detection)
                rate = self.capture_detection.get(cv2.CAP_PROP_FPS)  # 帧速率
                frame_num = self.capture_detection.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频文件的帧数
                self.frame_count_detection = frame_num
                duration = math.ceil(frame_num / rate)  # 视频总帧数/帧速率得出视频时长
                self.lb_cur_time_detection.setText('{:0>2d}:{:0>2d}'.format(0, 0))
                self.lb_max_time_detection.setText('{:0>2d}:{:0>2d}'.format(int(duration / 60), duration % 60))
                self.btn_play_pause_detection.setEnabled(True)
                self.btn_backward_detection.setEnabled(True)
                self.btn_forward_detection.setEnabled(True)
                self.slider_video_detection.setEnabled(True)
                self.btn_detection.setEnabled(True)
                self.media_player_detection.mediaStatusChanged.connect(self.handle_media_status_detection)

    def swap(self):
        """车牌替换实现，替换时弹出进度条，且不可操作"""
        des_video_dir = './data_des_swap'
        print("进入替换流程")
        self.thread_c = ThreadCover(self.file_name_swap, des_video_dir, self.model_yolo, self.model_crnn, 0.1)
        self.thread_c.processSignal.connect(self.showProgress_swap)  # 为进度条提供数据
        self.thread_c.finished.connect(self.swap_finished)  # 替换完后的操作
        self.thread_c.start()
        self.pd_swap = QProgressDialog(self)
        self.pd_swap.setWindowTitle("进度")
        self.pd_swap.setCancelButtonText("取消替换")
        self.swap_success = 1  # 是否成功替换（或被用户取消）
        self.pd_swap.canceled.connect(self.swap_terminate)
        self.pd_swap.setRange(0, 100)
        self.pd_swap.setValue(0)
        self.pd_swap.exec_()

    def showProgress_swap(self, msg):
        """进度展示"""
        val = (msg / int(self.frame_count_swap)) * 100  # 进度
        self.pd_swap.setValue(val)
        if val == 100:
            self.pd_swap.cancel()

    def swap_terminate(self):
        """替换被取消后的操作"""
        self.swap_success = 0
        self.thread_c.terminate()
        QMessageBox.information(self, '车牌替换', '替换终止！', QMessageBox.Yes, QMessageBox.Yes)

    def swap_finished(self):
        """替换完的操作"""
        if self.swap_success == 1:
            print("替换完成")
            QMessageBox.information(self, '车牌替换', '替换完成！', QMessageBox.Yes, QMessageBox.Yes)
            self.media_player_swap.setMedia(
                QMediaContent(QUrl.fromLocalFile('./data_des_swap/{}'.format(self.file_name_swap.split('/')[-1]))))
            self.media_player_swap.pause()

    def detection(self):
        """车牌篡改检测实现，替换时弹出进度条，且不可操作"""
        des_video_dir = './data_des_detection'
        print("进入检测流程")
        self.thread_d = ThreadImageTamperingDetection(self.file_name_detection, des_video_dir, self.model_yolo,
                                                      self.model_cnn)
        self.thread_d.processSignal.connect(self.showProgress_detection)  # 为进度条提供数据
        self.thread_d.finished.connect(self.detection_finished)  # 替换完后的操作
        self.thread_d.start()
        self.pd_detection = QProgressDialog(self)
        self.pd_detection.setWindowTitle("进度")
        self.pd_detection.setCancelButtonText("取消检测")
        self.detection_success = 1  # 是否成功检测（或被用户取消）
        self.pd_detection.canceled.connect(self.detection_terminate)
        self.pd_detection.setRange(0, 100)
        self.pd_detection.setValue(0)
        self.pd_detection.exec_()

    def showProgress_detection(self, msg):
        """进度展示"""
        val = (msg / self.frame_count_detection) * 100  # 进度
        self.pd_detection.setValue(val)
        if val == 100:
            self.pd_detection.cancel()

    def detection_terminate(self):
        """检测被取消后的操作"""
        self.detection_success = 0
        self.thread_d.terminate()
        QMessageBox.information(self, '车牌篡改检测', '检测终止！', QMessageBox.Yes, QMessageBox.Yes)

    def detection_finished(self):
        """检测完的操作"""
        if self.detection_success == 1:
            print("检测完成")
            QMessageBox.information(self, '车牌篡改检测', '检测完成！', QMessageBox.Yes, QMessageBox.Yes)
            self.media_player_detection.setMedia(
                QMediaContent(
                    QUrl.fromLocalFile('./data_des_detection/{}'.format(self.file_name_detection.split('/')[-1]))))
            self.media_player_detection.pause()



    def save_result(self):
        """结果保存"""
        if self.stackedWidget.currentIndex() == 0:
            # 替换页
            if self.swap_success == 1:
                # 保存车牌替换结果（需合并视频和音频）
                path, _ = QFileDialog.getSaveFileName(self, '保存结果', './', "视频文件(*.mp4;*.avi)")
                print("结果保存路径：", path)
                shutil.copyfile('./data_des_swap/{}'.format(self.file_name_swap.split('/')[-1]), path)
                self.swap_success = 0  # 等待下一个视频处理完之后才可以保存
                QMessageBox.information(self, '保存结果', '保存成功！', QMessageBox.Yes, QMessageBox.Yes)
            else:
                # 视频未处理
                QMessageBox.warning(self, '保存结果', '暂无已处理的视频！', QMessageBox.Yes, QMessageBox.Yes)
        elif self.stackedWidget.currentIndex() == 1:
            # 检测页
            if self.detection_success == 1:
                # 保存车牌替换结果（需合并视频和音频）
                path, _ = QFileDialog.getSaveFileName(self, '保存结果', './', "视频文件(*.mp4;*.avi)")
                print("结果保存路径：", path)
                shutil.copyfile('./data_des_detection/{}'.format(self.file_name_detection.split('/')[-1]), path)
                self.detection_success = 0  # 等待下一个视频处理完之后才可以保存
                QMessageBox.information(self, '保存结果', '保存成功！', QMessageBox.Yes, QMessageBox.Yes)
            else:
                # 视频未处理
                QMessageBox.warning(self, '保存结果', '暂无已处理的视频！', QMessageBox.Yes, QMessageBox.Yes)

    def forward_control(self, slider, media_player, step, duration):
        """控制前进step ms"""
        if slider.value() + step <= duration:
            slider.setValue(slider.value() + step)
            media_player.setPosition(media_player.position() + step)
        else:
            slider.setValue(duration)
            media_player.setPosition(duration)

    def backward_control(self, slider, media_player, step):
        """控制前进step ms"""
        if slider.value() - step >= 0:
            slider.setValue(slider.value() - step)
            media_player.setPosition(media_player.position() - step)
        else:
            slider.setValue(0)
            media_player.setPosition(0)

    def update_duration_swap(self, duration):
        """更新视频时长，以便设置slider的最大值"""
        self.slider_video_swap.setRange(0, duration)

    def update_position_swap(self, position):
        """更新slider的位置"""
        self.slider_video_swap.setValue(position)

    def set_video_position_swap(self, position):
        """拖动进度条滑块时改变视频播放进度"""
        self.media_player_swap.setPosition(position)

    def slider_pressed_swap(self, position):
        """单击进度条时改变视频播放进度"""
        self.media_player_swap.setPosition(position)

    def update_duration_detection(self, duration):
        """更新视频时长，以便设置slider的最大值"""
        self.slider_video_detection.setRange(0, duration)

    def update_position_detection(self, position):
        """更新slider的位置"""
        self.slider_video_detection.setValue(position)

    def set_video_position_detection(self, position):
        """拖动进度条滑块时改变视频播放进度"""
        self.media_player_detection.setPosition(position)

    def slider_pressed_detection(self, position):
        """单击进度条时改变视频播放进度"""
        self.media_player_detection.setPosition(position)

    def closeEvent(self, event):
        """重写关闭方法，实现在关闭前video.release()，防止内存泄漏"""
        if self.capture_swap is not None:
            self.capture_swap.release()
        if self.capture_detection is not None:
            self.capture_detection.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
