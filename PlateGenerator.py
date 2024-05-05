import numpy as np
import cv2

# 省份
provinces = {
    "京": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z'],
    "津": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z'],
    "冀": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'R', 'T'],
    "晋": ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'J', 'K', 'L', 'M'],
    "蒙": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'],
    "辽": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P'],
    "吉": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'],
    "黑": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R'],
    "沪": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z'],
    "苏": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N'],
    "浙": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L'],
    "皖": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S'],
    "闽": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'],
    "赣": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'],
    "鲁": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'U', 'V', 'Y'],
    "豫": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'U'],
    "鄂": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S'],
    "湘": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'S', 'U'],
    "粤": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y'],
    "桂": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R'],
    "琼": ['A', 'B', 'C', 'D', 'E', 'F'],
    "渝": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z'],
    "川": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
           'Z'],
    "贵": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'],
    "云": ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S'],
    "藏": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "陕": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'V'],
    "甘": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P'],
    "青": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    "宁": ['A', 'B', 'C', 'D', 'E'],
    "新": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']
}
provinces_idx = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
                 "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
                 "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
                 "甘", "青", "宁", "新"]
# 阿拉伯数字
nums = ['{}'.format(x) for x in range(10)]

# 英文字母（除I和O外）
# chr用于将数字转为对应的ASCII码，ord反之
letters = [chr(x + ord('A')) for x in range(26) if not chr(x + ord('A')) in ['I', 'O']]


def random_select_one(lst):
    """从给定的列表中随机选择一个元素"""
    return np.random.choice(lst, 1, replace=False)[0]


def generate_plate_num(plate_type=0, province='甘'):
    """
    随机生成车牌号
    plate_type：车牌号类型，0为绿牌，1为蓝牌
    province：省份
    """
    if plate_type == 0:
        # 绿牌：第一位序号使用L及以后的英文字母（除O外），其余5位使用数字
        res = province + random_select_one(provinces.get(province))
        letters_for_green = letters[letters.index('L'):]  # L及以后的英文字母（除O外）
        res += random_select_one(letters_for_green)
        for _ in range(5):
            # 生成数字
            res += random_select_one(nums)
    else:
        # 蓝牌：5位车牌号中使用3位英文字母（除I和O）或使用不存在的发牌机关代号，两种方式随机选择
        sel_num = np.random.randint(0, 2)
        if province in ['京', '津', '沪', '渝'] or sel_num == 0:
            # 5位车牌号中使用3位英文字母（除I和O）
            res = province + random_select_one(provinces.get(province))
            letter_idx = list(np.random.choice(5, 3, replace=False))  # 使用英文字母的位置
            for idx in range(5):
                if idx in letter_idx:
                    # 生成字母
                    res += random_select_one(letters)
                else:
                    # 生成数字
                    res += random_select_one(nums)
        else:
            # 使用不存在的发牌机关代号，setdiff1d用于求差集（前有后没有的元素）
            res = province + random_select_one(np.setdiff1d(letters, provinces.get(province)))
            letter_num = np.random.randint(0, 3)  # 英文字母的个数
            letter_idx = []
            if letter_num > 0:
                letter_idx = list(np.random.choice(5, letter_num, replace=False))  # 使用英文字母的位置
            for i in range(5):
                if i in letter_idx:
                    # 生成字母
                    res += random_select_one(letters)
                else:
                    # 生成数字
                    res += random_select_one(nums)
    return res


def get_char_location(plate_type=0):
    """
    获取各字符在车牌中的坐标
    plate_type：车牌号类型，0为绿牌，1为蓝牌
    """
    length = 8 if plate_type == 0 else 7  # 车牌字符数，7为蓝牌、8为绿牌
    char_location = np.zeros((length, 4), dtype=np.int32)  # [左上x，左上y，右上x，右上y]
    # x轴方向
    space_pro_char1 = 34 if plate_type == 1 else 49  # 发牌机关代号与第一个字符之间的间距
    space_char = 12 if plate_type == 1 else 9  # 各字符之间的间距
    width_char = 45  # 字符宽度（蓝牌所有字符宽度为45，绿牌省份简称字符宽度为45，其余字符宽度均为43，在下方会有更改）
    for i in range(length):
        if i == 0:
            char_location[i, 0] = 15  # 省份简称字符和车牌左端的间距（绿牌为15.5，舍入为15）
        elif i == 2:
            char_location[i, 0] = char_location[i - 1, 2] + space_pro_char1  # 字符左端坐标
        else:
            char_location[i, 0] = char_location[i - 1, 2] + space_char  # 字符左端坐标
        if length == 8 and i > 0:
            width_char = 43  # 绿牌非首字符宽度为43
        char_location[i, 2] = char_location[i, 0] + width_char  # 字符右端坐标
    # y轴方向
    char_location[:, 1] = 25  # 字符顶端坐标（字符顶端与车牌顶端间距 = 25）
    char_location[:, 3] = 115  # 字符底端坐标（字符底端与车牌顶端间距 = 25 + 90）
    return char_location


def generate_plate(plate_num, save_dir=""):
    """
    生成车牌图片
    :param plate_num: 车牌号
    :param save_dir: 目标文件夹，若为空，则不保存
    :return: 生成的车牌图片
    """
    plate_type = 0 if len(plate_num) == 8 else 1  # 车牌号类型，0为绿牌，1为蓝牌
    if plate_type == 0:
        # 绿牌
        char_location = get_char_location(plate_type)  # 字符坐标
        img_plate = cv2.imread('plate_bgs/green.png')
        for i in range(len(plate_num)):
            # 对单个字符逐个粘贴至车牌底牌
            s = plate_num[i]
            img_char = cv2.imread('./chars/green_' +
                                  ('p' + str(provinces_idx.index(s)) if '\u4e00' <= s <= '\u9fff' else s) + '.jpg')
            location = char_location[i]
            left_top = (location[0] * 2, location[1] * 2)  # 左上角坐标
            right_bottom = (location[2] * 2, location[3] * 2)  # 右下角坐标
            img_plate[left_top[1]: right_bottom[1], left_top[0]: right_bottom[0]] = np.where(
                img_char < 128, 0,
                img_plate[left_top[1]: right_bottom[1], left_top[0]: right_bottom[0]])
        if save_dir != "":
            cv2.imencode('.jpg', img_plate)[1].tofile(save_dir + '/' + plate_num + '.jpg')  # 保存文件，避免中文乱码
    else:
        # 蓝牌
        char_location = get_char_location(plate_type)  # 字符坐标
        img_plate = cv2.imread('plate_bgs/blue.png')
        for i in range(len(plate_num)):
            # 对单个字符逐个粘贴至车牌底牌
            s = plate_num[i]
            img_char = cv2.imread('./chars/blue_' +
                                  ('p' + str(provinces_idx.index(s)) if '\u4e00' <= s <= '\u9fff' else s) + '.jpg')
            location = char_location[i]
            left_top = (location[0] * 2, location[1] * 2)  # 左上角坐标
            right_bottom = (location[2] * 2, location[3] * 2)  # 右下角坐标
            img_plate[left_top[1]: right_bottom[1], left_top[0]: right_bottom[0]] = np.where(
                img_char < 128, 255,
                img_plate[left_top[1]: right_bottom[1], left_top[0]: right_bottom[0]])
        if save_dir != "":
            cv2.imencode('.jpg', img_plate)[1].tofile(save_dir + '/' + plate_num + '.jpg')  # 保存文件，避免中文乱码
    return img_plate


if __name__ == '__main__':
    # for _ in range(10):
    # plate_num = generate_plate_num(0, '鄂')  # 0绿1蓝
    # # res = generate_plate(plate_num, './plates_res')
    # res = generate_plate(plate_num, "")
    # cv2.imshow('', res)
    # cv2.waitKey(0)
    # print(get_char_location())
    generate_plate('甘JP10988', './plates_res')
