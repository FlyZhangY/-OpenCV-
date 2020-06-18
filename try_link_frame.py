"""
  f_height = f_shape[0]  # 原视频图片的高度
        f_width = f_shape[1]  # 代码中这点比较重要。
        video_writer = cv2.VideoWriter(save_path, four_cc, fps, (int(f_width), int(f_height)))
"""

import os
import cv2
import math


def read_video():
    """
    获取到输入的视频路径，并建立保存的路径。
    :return:
    """
    video_path = './video/zm.mp4'
    save_path = './video/1.avi'
    return video_path, save_path


def clip_video():
    """
    对视频任意时间段进行剪切
    :return:
    """
    video_path, save_path = read_video()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('video is not opened')
    else:
        success, frame = cap.read()
        f_shape = frame.shape
        f_height = f_shape[0]  # 原视频图片的高度
        f_width = f_shape[1]
        fps = cap.get(5)  # 帧速率
        print(fps)
        frame_number = cap.get(7)  # 视频文件的帧数
        print(frame_number)
        duration = frame_number / fps  # 视频总帧数/帧速率 是时间/秒【总共有多少秒的视频时间】
        print('请注意视频的总时间长度为 %s 秒' % str(duration))
        start = 45
        start_time = fps * float(start)
        end = 46
        end_time = fps * float(end)
        # AVI格式编码输出 XVID
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(save_path, four_cc, fps, (int(f_width), int(f_height)))
        num = 0
        while True:
            success, frame = cap.read()
            if (fps * 44 <= int(num) <= fps * 46 )or (fps * 5<=num<=fps * 6):
                if success:
                    video_writer.write(frame)
                else:
                    break
            num += 1
            if num > frame_number:
                break
        cap.release()


if __name__ == '__main__':
    clip_video()
