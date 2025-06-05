import os
import cv2
import pandas as pd
from numpy import matlib
from DBO import DBO
from DMO import DMO
from DOA import DOA
from Global_Vars import Global_Vars
from Image_Results import *
from Model_DenseNet import Model_DenseNet
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Model_MobileNet import Model_MobileNet
from Objfun import objfun
from Plot_Results import *
from Proposed import Proposed
from RSO import RSO


def read_Dataset(path_dir):
    in_dir = os.listdir(path_dir)
    Images = []
    Target = []
    for i in range(len(in_dir)):
        file_dir = path_dir + '/' + in_dir[i]
        file = os.listdir(file_dir)
        for j in range(len(file)):
            print(i, j)
            file_name = file_dir + '/' + file[j]
            data = cv2.imread(file_name)
            width = 512
            height = 512
            dim = (width, height)
            resized_image = cv2.resize(data, dim)
            Images.append(resized_image)
            Target.append(i)

    return Images, Target


# Read Dataset1
an = 0
if an == 1:
    out_dir = './Dataset'
    data, Target = read_Dataset(out_dir)

    np.save('Image.npy', data)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    np.save('Target.npy', target)

# Optimization for Object Detection and Classification
an = 0
if an == 1:
    Data = np.load('Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 4  # Hidden Neuron, step per Epoch(Yolov8 and GRU)
    xmin = matlib.repmat([5, 5, 5, 5], Npop, 1)
    xmax = matlib.repmat([255, 50, 255, 50], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun
    Max_iter = 50

    print("RSO...")
    [bestfit1, fitness1, bestsol1, time1] = RSO(initsol, fname, xmin, xmax, Max_iter)  # RSO

    print("DMO...")
    [bestfit4, fitness4, bestsol4, time4] = DMO(initsol, fname, xmin, xmax, Max_iter)  # DMO

    print("DBO...")
    [bestfit2, fitness2, bestsol2, time2] = DBO(initsol, fname, xmin, xmax, Max_iter)  # DBO

    print("DOA...")
    [bestfit3, fitness3, bestsol3, time3] = DOA(initsol, fname, xmin, xmax, Max_iter)  # DOA

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Best_Sol.npy', BestSol)

# Classification
an = 0
if an == 1:
    Feat = np.load('Image.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('Best_Sol.npy', allow_pickle=True)  # loading step
    EVAL = []
    Epoch = [100, 200, 300, 400, 500]
    for act in range(len(Epoch)):
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(act, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_GRU(Feat, Target, Epoch[act], sol)  # GRU With optimization
        Eval[5, :], pred1 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                           Epoch[act])  # Model DenseNet
        Eval[6, :], pred2 = Model_MobileNet(Train_Data, Train_Target, Test_Data,
                                            Test_Target, Epoch[act])  # Model AutoEncode
        Eval[7, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target,
                                       Epoch[act])  # Model LSTM
        Eval[8, :], pred4 = Model_GRU(Feat, Target, Epoch[act])  # GRU Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_Hidden.npy', EVAL)  # Save Eval all

plotConvResults()
plot_Result()
plot_seg_results()
Image_Results()
Sample_Images()
