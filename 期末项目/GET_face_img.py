import cv2
import numpy as np
import os
import glob

img_path = '.\\imge\\'      #带处理的图片存放的根目录
face_path = '.\\face_imge\\'        #处理后得到的人脸存放的路径
model = './model/haarcascade_frontalface_default.xml'
if not os.path.isfile(model):
    print("你缺少人脸检测模型——haarcascade_frontalface_default.xml")
if not os.path.exists(face_path):
	os.makedirs(face_path)
file_lists = []             #所有带处理的图片路径
# 获取文件夹下所有的jpg文件，将jpg文件路径保存到file_lists列表中
for root,dirs,files in os.walk(img_path):
    file_pattern=os.path.join(root,'*.jpg')
    for f in glob.glob(file_pattern):
        # jpg文件
        file_lists.append(f)
# 检测人脸函数
def detect(filename,outpath):
  #   调用人脸检测模型
  face_cascade = cv2.CascadeClassifier(model)
  img = cv2.imread(filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 人脸检测
  faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.12,
    minNeighbors = 5,
    minSize = (5,5)
  )
  img_name = filename.split('\\')[-1].split(".")[0]
  print(img_name)
  print(len(faces))
  i=1
  # (x,y,w,h) 用来确定人脸出现的位置 ： y:y+h,x:x+w
  for (x,y,w,h) in faces:
    img_face=img[y-1:y+h+1,x-1:x+w+1,:]     #范围加一的原因是，为了提取人脸周围更大一点的范围
    img_face_path =outpath+img_name+'_'+str(i)+'.jpg'   #保存人脸的命名方式为，原图像名_序号（由该张图片人脸个数决定）
    cv2.imwrite(img_face_path,img_face)
    i=i+1
# 循环提取每张图片的人脸并保存
for filename in file_lists:
    detect(filename,face_path)
