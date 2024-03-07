#!/usr/bin/env python
# coding=utf-8

import os
import cv2

# 视频路径保存路径
videos_path = r"E:\machine-learrning\ceshi1"
#图片保存路径，路径的最后一个名字是文件的名，同时生成一个多余的名字
images_path = r"E:\dataset\one\cam1"
# 如果不存在images_path创建，存在则提示报错
if not os.path.exists(images_path):
    os.makedirs(images_path)

# 遍历读取视频文件---支持多级目录下的视频文件遍历
i = 0
j = 1 #图片名对应数字，分批次的话，记得修改该参数，从25张图开始，就设置为25
file_count = 0 #提取的视频文件数量
# os.walk()函数遍历多个子目录下的视频文件，可以灵活应用于“多个子文件夹，多个视频文件”和“只有一个子文件夹，只有一个视频”的不同情况
for root, dirs, files in os.walk(videos_path):
    # os.mkdir(images_path + '/' + "sss")
    for file_name in files:
        file_count += 1
        i += 1
        # 以“日期+后缀”对文件夹命名
        # os.mkdir(images_path + '/' + '0324_%d' % i)
        # img_full_path = os.path.join(images_path, '0324_%d' % i) + '/'
        # 以“原视频名”对文件夹命名
        # os.mkdir(images_path + '/' + file_name.split('.')[0])
        # 视频的完整路径
        #img_full_path = os.path.join(images_path, file_name.split('.')[0]) + '/'
        img_full_path = os.path.join(images_path)
        # cv2.VideoCapture()根据路径读取视频文件
        videos_full_path = os.path.join(root, file_name)
        cap = cv2.VideoCapture(videos_full_path)
        print('\n开始处理第 ', str(i), ' 个视频：' + file_name)

        # 以指定帧数抽取图片并保存
        frame_count = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_count += 1
            # 修改帧数，1分钟等于1440，1秒为24帧，每个视频出第一秒的数据
            if frame_count%24 == 0:
                j+=1
                # 将抽取到的图片直接resize到指定的分辨率
                frame = cv2.resize(frame, (1275, 720))
                name = img_full_path + "0324_%d_%06d.jpg" % (i, frame_count)
                # 以“原视频名+帧数”对图片命名
                name = img_full_path + str(j) + ".jpg"
                # name ="a"+str(j)+ ".jpg"
                print(name)
                cv2.imwrite(name, frame)
                break
print('\n一共 ', str(file_count), ' 个视频,', '已全部处理完毕！')
