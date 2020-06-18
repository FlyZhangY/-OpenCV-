 

#  项目始于对OpenCV学习的时候所做，主要功能及其目的如下

​                                                             

 

 

**基于OpenCV的视频人物查找及剪辑**

 

## 小组分工

| **名称** | **基于OpenCV的视频人物查找及剪辑** |                |                |                |
| -------- | ---------------------------------- | -------------- | -------------- | -------------- |
| **学部** | **工学部**                         |                |                |                |
| **班级** | **2017****级大数据创新班**         |                |                |                |
| **姓名** | **韩凤丽**                         | **王晓玉**     | **张帅鹏**     | **张飞翔**     |
| **学号** | **1701190004**                     | **1701541034** | **1701570002** | **1701142037** |

 

## 指导老师



| **学院**                                                     | **大数据与智能技术学院** | **中心**     |            |
| ------------------------------------------------------------ | ------------------------ | ------------ | ---------- |
| **课程编号**                                                 |                          | **课程名称** |            |
| **课程类别**                                                 |                          | **任课教师** | **秦记峰** |
| **教师评语：**                                               |                          |              |            |
| **成绩评定：**  **分**     **任课教师签名：**         **年**   **月** **日** |                          |              |            |

**
**

**目录**

[一、 系统概述................................................................................................... 1](#_Toc10575)

[二、 系统现状和存在的问题........................................................................ 2](#_Toc25774)

[三、 关键算法................................................................................................... 3](#_Toc12714)

[3.1 Cascade级联分类器.......................................................................... 3](#_Toc11672)

[3.2特征脸算法EigenFace...................................................................... 3](#_Toc12443)

[四、 系统功能................................................................................................... 4](#_Toc9437)

[4.1 系统准备............................................................................................. 4](#_Toc27431)

[4.2 训练数据准备.................................................................................... 4](#_Toc7747)

[4.3视频分帧，获取图片........................................................................ 4](#_Toc25971)

[4.31系统实现思路.......................................................................... 4](#_Toc22284)

[4.32系统实现代码及注释............................................................ 5](#_Toc12982)

[4.33系统实现效果截图及分析................................................... 6](#_Toc15494)

[4.4人脸识别并提取................................................................................. 6](#_Toc32283)

[4.41设计算法模型：Cascade级联分类器............................... 6](#_Toc15360)

[4.42系统实现思路.......................................................................... 6](#_Toc23383)

[4.43系统实现代码及注释............................................................ 7](#_Toc3951)

[4.44系统实现效果截图及分析................................................... 8](#_Toc9530)

[4.5人脸模型训练..................................................................................... 8](#_Toc21587)

[4.51设计模型算法：特征脸算法EigenFace........................... 8](#_Toc8585)

[4.52系统实现思路.......................................................................... 8](#_Toc26260)

[4.53系统实现代码及注释............................................................ 9](#_Toc7390)

[4.54系统实现效果截图及分析................................................. 11](#_Toc1901)

[4.6 模型测试，提取相关人物全部视频信息................................ 11](#_Toc1642)

[4.61算法模型——特征脸算法及4.3训练出的模型......... 11](#_Toc22695)

[4.62系统实现思路........................................................................ 12](#_Toc4801)

[4.63系统实现代码及注释.......................................................... 12](#_Toc14000)

[4.64系统实现效果截图及分析................................................. 15](#_Toc26143)

[五、 总结.......................................................................................................... 16](#_Toc4171)

[参考文献............................................................................................................ 16](#_Toc2507)



**
**

**基于OpenCV的视频人物查找及剪辑

**摘要：**本项目是基于OpenCV的人脸识别技术，将一个完整的视频中自己想要了解的某个具体的人物片段，通过将视频分割成帧并且训练需要提取人物的面部特征，通过主成分分析算法进行人脸模型清洗并通过特征脸算法进行人脸识别，将该人物从视频中识别出来并将他的片段提取出来，组合成专属个人的视频剪辑，项目完成后，本产品可以使人们节省大量的时间从大量的视频片段中快速的了解自己想要了解的人物，还可以应用在监控的锁定人物中，就如《速度与激情》影片中通过天眼来快速找到某人的效果一样，快速的查找到某个人在一段时间内的全部活动。

**关键词：**人脸识别、关键帧提取、面部特征

 

**Abstract:** This project is based on the Face Recognition technology of OpenCV, which divides a complete video into frames and trains to extract the facial features of the characters. The Principal Component Analysis Algorithm is used to clean the face model and the Face Recognition is carried out through the Eigenface Algorithm. The character is recognized from the video and his fragments are extracted. Combined into a personal video clip, after the completion of the project, this product can save a lot of time for people to quickly understand the people appeared in video that they want to know from a large number of video clips, but also can be applied to monitor the lock-in person In things, just as Fast & Furious can quickly find someone through the eye of heaven,and quickly find out all the activities of someone over a period of time.

 

# Keywords: Face Recognition, Key Frame Extraction, Facial Features



# 一、系统概述

该系统主要是通过输入一段视频，然后把视频分割成帧，将每张视频图片进行人脸采集，通过面部训练，提取面部特征，与外部实际人物进行匹配，并将想要匹配的人的视频帧进行组合，形成只有匹配人物的视频组合。

结构图：

  ![无标题](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)  

 

**
**

# 二、系统现状和存在的问题

随着社会的不断进步以及各方面对于快速有效的自动身份验证的迫切要求，生物特征的识别技术在近几十年中得到了飞速的发展。作为人的一种内在的属性，并且具有很强的自身稳定性及个体的差异性，当前的生物特征识别技术多中多样。其中包括指纹识别、虹膜技术、步态识别；与这几种生物识别来说人脸识别由于更直接，友好，方便，更易为用户所接受。本产品旨在解决处理大量繁琐视频的主要应用片段，由于现在正处于信息化社会，人们一天中会接受到大量的视频信息，但真正符合自己所需的片段却少之又少，通过本产品，我们可以在大量的视频信息中提取到自己想要了解的人物片段，通过对关键帧的提取、训练、组合成一个具体的所需片段，可以使我们在短时间内准确的获取了解对象的一系列活动。

本产品适用于视频软件中提取自己喜欢的人物片段的提取，可以应用于监控视频的人物定位，准确的提取目标人物在一段时间内的活动，还可以在警方办案中使警方在大量的视频片段中快速的了解嫌疑人的活动动态。目前视频中人物的识别多用在视频播放平台如爱奇艺、腾讯，但是多是仅将视频中人物识别出来，即人物第一次出现就结束，并不针对该人出现的所有时间。或者人物侧颜有可能不能很准确的识别出来，所以存在一定的局限性。



# 三、关键算法

## 3.1 Cascade级联分类器

Cascade级联分类器是一种快速简单的分类方法，可以理解为将N个单类的分类器串联起来。如果一个事物能属于这一系列串联起来的的所有分类器，则最终结果就是 是，若有一项不符，则判定为否。比如人脸，它有很多属性，我们将每个属性做一成个分类器，如果一个模型符合了我们定义的人脸的所有属性，则我们人为这个模型就是一个人脸。cv::CascadeClassifier通过load()成员函数读取分类器，然后就可以使用detectMultiScale()成员函数进行多尺度的级联分类检测。

## 3.2特征脸算法EigenFace

EigenFace(特征脸)在人脸识别历史上应该是具有里程碑式意义的，其被认为是第一种有效的人脸识别算法。1987年 Sirovich and Kirby 为了减少人脸图像的表示（降维）采用了PCA（主成分分析）的方法，1991年 Matthew Turk和Alex Pentland首次将PCA应用于人脸识别，即将原始图像投影到特征空间，得到一系列降维图像，取其主元表示人脸，因其主元有人脸的形状，估称为“特征脸”。

EigenFace是一种基于统计特征的方法，将人脸图像视为随机向量，并用统计方法辨别不同人脸特征模式。EigenFace的基本思想是，从统计的观点，寻找人脸图像分布的基本元素，即人脸图像样本集协方差矩阵的特征向量，以此近似的表征人脸图像，这些特征向量称为特脸。



# 四、系统功能

## 4.1 系统准备

```
①先在该目录下创建video文件夹，来存放需要提取的视频
②在该目录下创建imge文件夹，用来存放视频中提取的图片
③在该目录下创建face_imge文件夹，用来存放视频中的人脸图片
④在该目录下创建model文件用来存放训练模型
⑤在该目录下创建face_train文件夹，用来存放训练集的图片
```

## 4.2 训练数据准备

在face_train文件夹，用来存放训练集的图片，每一个人的图片存放在同一个文件夹下，第一个文件夹不要放准备查找的人物图片，防止查找为空时返回0与监测结果发生歧义，可以多准备几人的照片当做训练集可以更高的提高训练的准确性。

如图：

```
![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)
```

文件夹1存放的目标检测人物图片

```
![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)
```

 

## 4.3视频分帧，获取图片

### 4.31系统实现思路

```
定义此函数video_split，目实是对视频文件切割成帧，并将每两帧保存一张图片，首先定义两个参数video_path:视频路径，save_path:保存切分后帧的路径，函数里设计步骤：
①读取视频
②设置帧速率
③以及定义每2帧保存一帧图片
④计算该帧出现的时间，并将该帧图片保存
⑤最后调用函数，输入视频路径即可。
```

### 4.32系统实现代码及注释

  import cv2 #导入opencv模块   import os   import time      def video_split(video_path,save_path):     '''     对视频文件切割成帧，并将每两帧保存一张图片     '''     '''     video_path:视频路径     save_path:保存切分后帧的路径     '''     vc=cv2.VideoCapture(video_path)     fps = vc.get(5) # 帧速率     #视频总帧数/帧速率  是时间/秒【总共有多少秒的视频时间】      #帧数/帧速率 是该帧出现的时间（单位：秒）     c=0     if vc.isOpened():      rval,frame=vc.read()     else:      rval=False     while rval:      rval,frame=vc.read()      # 每2帧保存一帧图片      if c % 2 == 0:        duration = c / fps  #该帧出现的时间（单位：秒）        duration = duration*100        # 将该帧图片保存        cv2.imwrite(save_path +  "/" + str('%05d'%duration)+'.jpg',frame)        cv2.waitKey(1) #1毫秒后进行下一步      c=c+1      DATA_DIR = "./video/zm.mp4" #视频数据主目录   SAVE_DIR = "./imge" #帧文件保存目录   print("正在处理视频文件",DATA_DIR)   # 对视频数据进行提取图片   video_split(DATA_DIR,SAVE_DIR)     

### 4.33系统实现效果截图及分析

```
文件夹下帧图片：
![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)
```

 

## 4.4人脸识别并提取

### 4.41设计算法模型：Cascade级联分类器

### 4.42系统实现思路

设置一个循环，检测每一张人脸，识别出人脸位置，进行提取

①加载分类器模型：

·从文件中加载级联分类器

·检测级联分类器是否被加载

·从文件中加载级联分类器

·从FileStorage节点读取分类器

②进行人脸检测

③确定人脸出现的位置提取并保存

注：我们直接调用模型。

### 4.43系统实现代码及注释

  import cv2   import numpy as np   import os   import glob      img_path = '.\\imge\\'   #带处理的图片存放的根目录   face_path = '.\\face_imge\\'    #处理后得到的人脸存放的路径   file_lists = []       #所有带处理的图片路径   for root,dirs,files in os.walk(img_path):     file_pattern=os.path.join(root,'*.jpg')     for f in glob.glob(file_pattern):       # jpg文件       file_lists.append(f)   # 检测人脸函数   def detect(filename,outpath):    #   调用人脸检测模型    face_cascade =  cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')    img = cv2.imread(filename)    gray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)    # 人脸检测    faces =  face_cascade.detectMultiScale(     gray,     scaleFactor = 1.12,     minNeighbors = 5,     minSize = (5,5)    )    img_name =  filename.split('\\')[-1].split(".")[0]    print(img_name)    print(len(faces))    i=1    # (x,y,w,h) 用来确定人脸出现的位置 ： y:y+h,x:x+w    for (x,y,w,h) in faces:       img_face=img[y-1:y+h+1,x-1:x+w+1,:]    #范围加一的原因是，为了提取人脸周围更大一点的范围     img_face_path  =outpath+img_name+'_'+str(i)+'.jpg'  #保存人脸的命名方式为，原图像名_序号（由该张图片人脸个数决定）     cv2.imwrite(img_face_path,img_face)     i=i+1   # 循环提取每张图片的人脸并保存   for filename in file_lists:     detect(filename,face_path)     

### 4.44系统实现效果截图及分析

识别的人脸图片保存在face_image文件夹下：

  ![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)  

## 4.5人脸模型训练

### 4.51设计模型算法：特征脸算法EigenFace

### 4.52系统实现思路

基本的思路：识别——训练——保存模型

```
①进行人脸模型训练,准备好训练集图片
②训练集第一个文件夹随便存放图片,不做目标训练,将目标人物训练集只要不放在第一个文件夹即可,
因为：第几个文件夹,它所产生的lable便是多少，但当预测为无时，仍返回0，所以对结果有影响
③将训练好的模型存放
```

### 4.53系统实现代码及注释

  import  os   import sys   import cv2   import numpy as np   def normalize(X, low, high, dtype=None):     """将X中的给定数组调整为介于高低之间的值"""     X = np.asarray(X)     minX, maxX = np.min(X), np.max(X)     # 归一化0到1.     X = X - float(minX)     X = X / float((maxX - minX))     # 规划到最低到最高之间.     X = X * (high-low)     X = X + low     if dtype is None:       return np.asarray(X)     return np.asarray(X, dtype=dtype)   def read_images(path, sz=None):     c = 0     X,y = [], []     for dirname, dirnames, filenames in  os.walk(path):       for subdirname in dirnames:         subject_path =  os.path.join(dirname, subdirname)         print(subject_path)         #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表         for filename in  os.listdir(subject_path):           try:             #合并目录，filepath指的就是\OPENCV\pycv-master\data\at\cy这样完整的路径             filepath = os.path.join(subject_path,  filename)             #im是图像文件             im =  cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)             im =  cv2.resize(im,(133,133))             if (im is None):                print ("image " + filepath  + " is none" )             # resize to given  size (if given)             if (sz is not  None):               im =  cv2.resize(im, sz)               #用一个im副本代替指向X             X.append(np.asarray(im,  dtype=np.uint8))             y.append(c)           except IOError as e:             print("I/O  error({0}): {1}").format(e.errno, e.strerror)           except:               print("Unexpected error:"), sys.exc_info()[0]             raise         c = c+1     return [X,y]      if __name__ == "__main__":     out_dir = None     [X,y] =  read_images('./face_train/')     # print(y)     #y看有几个组     y = np.asarray(y, dtype=np.int32)     # out_dir = './'     '''     文件夹a0不具备预测数据提供     cv2中face子模块目前支持的算法有:       （1）主成分分析（PCA）——Eigenfaces（特征脸）——函数：cv2.face.EigenFaceRecognizer_create（）   PCA：低维子空间是使用主元分析找到的，找具有最大方差的哪个轴。   缺点：若变化基于外部（光照），最大方差轴不一定包括鉴别信息，不能实行分类。       （2）线性判别分析（LDA）——Fisherfaces（特征脸）——函数： cv2.face.FisherFaceRecognizer_create()   LDA:线性鉴别的特定类投影方法，目标：实现类内方差最小，类间方差最大。       （3）局部二值模式（LBP）——LocalBinary Patterns Histograms——函数：cv2.face.LBPHFaceRecognizer_create()     '''     model =  cv2.face.EigenFaceRecognizer_create()     #人脸识别器     # print(np.shape(X))     # 进行训练     model.train(np.asarray(X),  np.asarray(y))     # 保存模型       model.save("./model/face_model_trained.xml")     

### 4.54系统实现效果截图及分析

最后训练出的模型保存在model文件夹下：

  ![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)  

## 4.6 模型测试，提取相关人物全部视频信息

### 4.61算法模型——特征脸算法及4.3训练出的模型

特征脸算法介绍见4.3，以及涉及到的模型为4.3训练得到的 。

### 4.62系统实现思路

```
①获取所有带测试的图片路径
②创建人脸识别器
③载入4.3的模型
④预测
⑤识别出是那个人物
⑥提取人物在视频中出现时间
⑦判断开始时间和结束时间，如果该人物出现的时差不于2s这里就表明，该人物在这段时间内连续存在
⑧最后将该任务出现的视频片段都提取出来
```

### 4.63系统实现代码及注释

  import cv2   import numpy as np   import os   import glob   # 切割视频函数   def clip_video(video_path,save_path,img_time):     """     对视频任意时间段进行剪切     :return:     """     cap = cv2.VideoCapture(video_path)     if not cap.isOpened():       print('video is not opened')     else:       success, frame = cap.read()       f_shape = frame.shape       f_height = f_shape[0] # 原视频图片的高度       f_width = f_shape[1]       fps = cap.get(5) # 帧速率       frame_number = cap.get(7) # 视频文件的帧数       duration = frame_number /  fps # 视频总帧数/帧速率  是时间/秒【总共有多少秒的视频时间】       print('请注意视频的总时间长度为 %s 秒' % str(duration))       # AVI格式编码输出 XVID         four_cc = cv2.VideoWriter_fourcc(*'XVID')       # 确定保存格式       video_writer =  cv2.VideoWriter(save_path, four_cc, fps, (int(f_width), int(f_height)))       num = 0       while True:         success, frame = cap.read()         # num/fps 结果是该帧出现的时间（单位 s)*10 是为了提高提取视频的精确到，以精确到0.1s         if int(10*num/fps) in  img_time:           if success:               video_writer.write(frame)           else:             break         num += 1         if num > frame_number:           break       cap.release()   # 获取所有带测试的图片路径   def get_img_file(path):     imgs_file = [] #所有图片的路径     for dirname, dirnames, filenames in  os.walk(path):       for file in filenames:         imgs_file.append(file)     return imgs_file   # 测试图片存放的根路径   path = './face_imge'   imgs_file = get_img_file(path)   # 获取所有带测试的图片路径   #创建人脸识别器   model = cv2.face.EigenFaceRecognizer_create()   # 载入模型   model.read("./model/face_model_trained.xml")   img_time = []      #人物出现的时间*10 提高剪切视频文件的准确率   go_out_time = []    #人物出现的持续时间，开始到结束[(开始，结束),()....]   start_time = 0   n_ms = 10000   for img in imgs_file:     bb = cv2.imread(path+'/'+img,  cv2.IMREAD_GRAYSCALE)     bb = cv2.resize(bb,(133,133))     # 预测     [p_label, p_confidence] =  model.predict(np.asarray(np.asarray(bb)))     #识别出来哪个人，p_label决定选择那个人     if (p_label==1 and  p_confidence<3000):          img =  img.split('.')[0].split('_')[0] #提取出图片名，确定时间信息       img_n = int(img)       img_time.append(img_n//10) #确定时间 精确到0.1秒       # 判断开始时间和结束时间，如果该人物出现的时差不于2s这里就表明，该人物在这段时间内连续存在       if start_time==0:         start_time = img_n/100       if (img_n/100)-n_ms>=2:         end_time = n_ms           go_out_time.append((start_time,end_time))         start_time = 0         n_ms = 1000       else:         n_ms = img_n/100   go_out_time.append((start_time,n_ms))   img_time=set(img_time)   for time in go_out_time:     print("该人物出现时间在%d~%d秒"%(time))   # 决定是否提取视频   print("是否将该人物视频提取出来？")   get_frame = int(input("是 输入 1，否 输入 0 ："))   if(get_frame == 1):     clip_video('./video/zm.mp4','./video/1.avi',img_time)     print("视频1.avi保存完成")     

 

### 4.64系统实现效果截图及分析

根据4.3训练出的模型，进行预测，最后将人物在视频中出现的所有片段进行提取保存。输入1，即视频保存为1.avi.

  ![img](file:///C:/Users/张飞翔~1/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)  

 

**
**

# 五、总结

基于opencv人脸识别视频剪辑系统的实现，完成了实质性的需求，即在海量视频中快速提取我们想要了解的某个人物的所有视频信息，从而满足客户需求，或者可用于帮助警方管理监控视频快速找到罪犯在监控视频出现的时间，以便于对犯罪分子进行快速定位。

整个项目基于视频分帧所得图片，进行人脸监测识别，保存识别后的图片，进行训练，建立模型，从而应用到视频中，最后提取模型所涉及人物的所有视频信息。最后训练的模型不是太准确，还需进行进一步对人脸识别版块进行优化，其他版块无需改动，如果人脸识别加上身形识别、步态识别、声音识别，那么用于影视快速作品快速视频剪辑，人物提取会起到很便利的作用，或者对警方通过监控来定位犯罪分子也会起到很好的帮助作用。

```
获取视频总帧数：cap.get(7)
获取帧速率：vc.get(5)
默认视频播放时长：视频总帧数/帧速率 =  cap.get(7)/ vc.get(5)
```

# 参考文献

[1]王伟,张佑生,方芳.人脸检测与识别技术综述[J].2006-05.
 [2]李刚，高政.人脸自动识别方法综述、计算机应用研究.2003.

[3]高建坡.视频序列中的人脸检测与跟踪算法研究.东南大学博士学位论文.2007-03

 

 

 

 

 

 

 

 

 

 

 

 