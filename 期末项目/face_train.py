'''
    1,进行人脸模型训练,准备好训练集图片
    2,训练集第一个文件夹随便存放图片,不做目标训练,将目标人物训练集只要不放在第一个文件夹即可,
    因为：第几个文件夹,它所产生的lable便是多少，但当预测为无时，仍返回0，所以对结果有影响
    3，将训练好的模型存放
'''
import os
import sys
import cv2
import numpy as np



def normalize(X, low, high, dtype=None):
    """将X中的给定数组调整为介于高低之间的值"""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # 归一化0到1.
    X = X - float(minX)
    X = X / float((maxX - minX))
    # 规划到最低到最高之间.
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)
def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print(subject_path)
            #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            for filename in os.listdir(subject_path):
                try:
                    #合并目录，filepath指的就是\OPENCV\pycv-master\data\at\cy这样完整的路径
                    filepath = os.path.join(subject_path, filename)
                    #im是图像文件
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    im = cv2.resize(im,(133,133))   #设置大小，根据实际情况设定
                    if (im is None):
                        print ("image " + filepath + " is none" )
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                        #用一个im副本代替指向X
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print("I/O error({0}): {1}").format(e.errno, e.strerror)
                except:
                    print("Unexpected error:"), sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

if __name__ == "__main__":
    out_dir = None
    [X,y] = read_images('./face_train/')
    # print(y)
    #y看有几个组
    y = np.asarray(y, dtype=np.int32)
    # out_dir = './'
    '''
    文件夹a0不具备预测数据提供
    cv2中face子模块目前支持的算法有:
        （1）主成分分析（PCA）——Eigenfaces（特征脸）——函数：cv2.face.EigenFaceRecognizer_create（）
PCA：低维子空间是使用主元分析找到的，找具有最大方差的哪个轴。
缺点：若变化基于外部（光照），最大方差轴不一定包括鉴别信息，不能实行分类。
        （2）线性判别分析（LDA）——Fisherfaces（特征脸）——函数： cv2.face.FisherFaceRecognizer_create()
LDA:线性鉴别的特定类投影方法，目标：实现类内方差最小，类间方差最大。
        （3）局部二值模式（LBP）——LocalBinary Patterns Histograms——函数：cv2.face.LBPHFaceRecognizer_create()
    '''
    model = cv2.face.EigenFaceRecognizer_create()
    #人脸识别器
    # print(np.shape(X))
    # 进行训练
    model.train(np.asarray(X), np.asarray(y))
    # 保存模型
    model.save("./model/face_model_trained.xml")

