import os
import numpy as np
from numpy import shape
from os import listdir
from numpy import tile
import operator
def knn(k,testdata,traindata,labels):
    traindatasize = traindata.shape[0]
    dif = tile(testdata,(traindatasize,1)) - traindata
    powdif = dif**2
    sumdif = powdif.sum(axis = 1)
    sqrdif = sumdif**0.5
    sortdistance=sqrdif.argsort()
    count={}
    for i in range(0,k):
        vote = labels[sortdistance[i]]
        count[vote]=count.get(vote, 0)+1
    sortcount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortcount[0][0]

def sortlabels(fname):
    filestr = fname.split(".")[0]
    labels = int(filestr.split("_")[0])
    return labels
# 将数据转换为数组
def datatoarray(fname):
    arr = []
    fh = open(fname)
    for i in range(0, 30):
        thisline = fh.readline()
        for j in range(0,30):
            arr.append(int(thisline[j]))
    return arr

def traindata():
    labels = []
    trainfile = listdir("txt")
    num = len(trainfile)
    trainarr= np.zeros((num, 900))
    for i in range(0,num):
        thisfname = trainfile[i]
        thislabel = sortlabels(thisfname)
        labels.append(thislabel)
        trainarr[i] = datatoarray("./txt/"+thisfname)
    return trainarr, labels

def datatest():
    trainarr,labels = traindata()
    testlist = listdir("txt1")
    tnum=len(testlist)
    count = 0
    for i in range(tnum):
        thisname = testlist[i]
        testarr = datatoarray("txt1/"+thisname)
        rknn = knn(3, testarr, trainarr, labels)
        print(str(thisname)+' : '+str(rknn))
        if(rknn == int(thisname.split(".")[0].split("_")[0])):
            count = count + 1
        count = float(count)
        tnum = float(tnum)
        a = count/tnum
    print(a)
datatest()
