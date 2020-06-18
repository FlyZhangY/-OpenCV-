import cv2 #导入opencv模块
import os
import time

def video_split(video_path,save_path):
	'''
	对视频文件切割成帧，并将每两帧保存一张图片
	'''
	'''
	 video_path:视频路径
	 save_path:保存切分后帧的路径
	'''
	vc=cv2.VideoCapture(video_path)
	fps = vc.get(5)  # 帧速率
	#视频总帧数/帧速率 是时间/秒【总共有多少秒的视频时间】
	#帧数/帧速率  是该帧出现的时间（单位：秒）
	c=0
	if vc.isOpened():
		rval,frame=vc.read()
	else:
		rval=False
	while rval:
		rval,frame=vc.read()
		# 每2帧保存一帧图片
		if c % 2 == 0:
			duration = c / fps		#该帧出现的时间（单位：秒）
			duration = duration*100
			# 将该帧图片保存
			cv2.imwrite(save_path + "/" + str('%05d'%duration)+'.jpg',frame)
			cv2.waitKey(1)	#1毫秒后进行下一步
		c=c+1

DATA_DIR = "./video/zm.mp4" #视频数据主目录
SAVE_DIR = "./imge" #帧文件保存目录
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)
print("正在处理视频文件",DATA_DIR)
# 对视频数据进行提取图片
video_split(DATA_DIR,SAVE_DIR)
