import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

plate_chr = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


class MyNet(nn.Module):
    """自定义CRNN模型"""

    def __init__(self, cfg=None, num_classes=78, export=False):
        super(MyNet, self).__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.loc(x)
        x = self.newCnn(x)
        if self.export:
            conv = x.squeeze(2)
            conv = conv.transpose(2, 1)
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2)
            conv = conv.permute(2, 0, 1)
            output = F.log_softmax(conv, dim=2)
            return output


def decode_plate(codes):
    pre = 0
    res_codes = []
    for i in range(len(codes)):
        if codes[i] != 0 and codes[i] != pre:
            res_codes.append(codes[i])
        pre = codes[i]
    return res_codes


def image_processing(img, device, img_size):
    """归一化图像矩阵"""
    img_h, img_w = img_size
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, img_size):
    """推理得出车牌号编码并解码"""
    input = image_processing(img, device, img_size)
    res = model(input)  # 模型得出的若干字符编码置信度矩阵
    codes_tensor = res.argmax(dim=2)  # 得出置信度最大的字符编码
    codes = codes_tensor.view(-1).detach().cpu().numpy()  # 转换为numpy数组
    res_codes = decode_plate(codes)  # 过滤
    plate = ""
    for i in res_codes:
        # 车牌号解码
        plate += plate_chr[int(i)]
    return plate


def init_model(device, model_path):
    """
    初始化模型
    :param device: cpu or gpu
    :param model_path: 模型文件路径
    :return: 模型对象
    """
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = MyNet(num_classes=len(plate_chr), export=True, cfg=cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def predict(device, model, img):
    """
    推理
    :param device: cpu or gpu
    :param model: 推理模型
    :param img: 图像数据
    :return: 单车牌号
    """
    img_h = 48
    img_w = 168
    img_size = (img_h, img_w)
    plate = get_plate_result(img, device, model, img_size)
    return plate


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/crnn.pth'
    model = init_model(device, model_path)
    image_path = './test_plates/Snipaste_2024-03-25_15-49-02.jpg'
    img = cv2.imread(image_path)
    print(predict(device, model, img))
