import cv2
import numpy as np
import os
import glob

# 切割视频函数
def clip_video(video_path,save_path,img_time):
    """
    对视频任意时间段进行剪切
    :return:
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('video is not opened')
    else:
        success, frame = cap.read()
        f_shape = frame.shape
        f_height = f_shape[0]  # 原视频图片的高度
        f_width = f_shape[1]
        fps = cap.get(5)  # 帧速率
        frame_number = cap.get(7)  # 视频文件的帧数
        duration = frame_number / fps  # 视频总帧数/帧速率 是时间/秒【总共有多少秒的视频时间】
        print('请注意视频的总时间长度为 %s 秒' % str(duration))
        # AVI格式编码输出 XVID
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        # 确定保存格式
        video_writer = cv2.VideoWriter(save_path, four_cc, fps, (int(f_width), int(f_height)))
        num = 0
        while True:
            success, frame = cap.read()
            # num/fps 结果是该帧出现的时间（单位 s)*10 是为了提高提取视频的精确到，以精确到0.1s
            if int(10*num/fps) in img_time:
                if success:
                    video_writer.write(frame)
                else:
                    break
            num += 1
            if num > frame_number:
                break
        cap.release()
# 获取所有带测试的图片路径
def get_img_file(path):
    imgs_file = []  #所有图片的路径
    for dirname, dirnames, filenames in os.walk(path):
        for file in filenames:
           imgs_file.append(file)
    return imgs_file
# 测试图片存放的根路径
path = './face_imge'

imgs_file = get_img_file(path)      # 获取所有带测试的图片路径
#创建人脸识别器
model = cv2.face.EigenFaceRecognizer_create()
# 载入模型
model.read("./model/face_model_trained.xml")
img_time = []           #人物出现的时间*10 提高剪切视频文件的准确率
go_out_time = []        #人物出现的持续时间，开始到结束[(开始，结束),()....]
start_time = 0
n_ms = 10000
for img in imgs_file:
    bb = cv2.imread(path+'/'+img, cv2.IMREAD_GRAYSCALE)
    bb = cv2.resize(bb,(133,133))#设置大小，根据实际情况设定
    # 预测
    [p_label, p_confidence] = model.predict(np.asarray(np.asarray(bb)))
    #识别出来哪个人，p_label决定选择那个人
    if (p_label==1 and p_confidence<3000):

        img = img.split('.')[0].split('_')[0] #提取出图片名，确定时间信息
        img_n = int(img)
        img_time.append(img_n//10)  #确定时间 精确到0.1秒
        # 判断开始时间和结束时间，如果该人物出现的时差不于2s这里就表明，该人物在这段时间内连续存在
        if start_time==0:
            start_time = img_n/100
        if (img_n/100)-n_ms>=2:
            end_time = n_ms
            go_out_time.append((start_time,end_time))
            start_time = 0
            n_ms = 1000
        else:
            n_ms = img_n/100
go_out_time.append((start_time,n_ms))
img_time=set(img_time)
for time in go_out_time:
    print("该人物出现时间在%d~%d秒"%(time))
# 决定是否提取视频
print("是否将该人物视频提取出来？")
get_frame = int(input("是 输入 1，否 输入 0 ："))
if(get_frame == 1):
    clip_video('./video/zm.mp4','./video/1.avi',img_time)
    print("视频zm1.avi保存完成")
