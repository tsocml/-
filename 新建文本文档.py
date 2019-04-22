import numpy as np
import pandas as pd
import operator
from glob import glob
from PIL import Image
import os
from os import listdir
a = glob(r'Generate_handwritten_number-master\data_pngs\Num_1\*.png')
def picture2code(filename1,filename2):
    image_file = Image.open(filename1)
    # 转成黑白图像
    image_file = image_file.convert('1')
    width,height=image_file.size
    f1=open(filename1,'r')
    f2=open(filename2,'w')
    for i in range(height):
        for j in range(width):
            # 获取每个像素值
            pixel=int(image_file.getpixel((j,i))/255)
            # 黑白图像中0代表黑色，1代表白色
            # 我希望有内容的部分表示为1，所以将0和1互换
            if(pixel==0):
                pixel=1
            elif(pixel==1):
                pixel=0
            f2.write(str(pixel))
            if(j==width-1):
                # 换行
                f2.write('\n')
    f1.close()
    f2.close()

path_picture='Num_9'
path_txt='txt'
# 文件夹下所有文件
pictureList=listdir(path_picture)
m=len(pictureList)
for i in range(m):
    pictureNameStr=pictureList[i]
    # 图像路径的完整表示
    picturelocation=os.path.join(path_picture, pictureNameStr)
    # 获取文件前缀，即文件名
    pictureStr=pictureNameStr.split('.')[0]
    # 生成的文本路径的完整表示
    txtlocation=os.path.join(path_txt, '%s.txt'%pictureStr)
    picture2code(picturelocation,txtlocation)
