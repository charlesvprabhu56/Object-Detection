import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_Vars
from Model_GRU import Model_GRU
from Model_yolov8_GRU import Model_Yolov8


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Image = Model_Yolov8(data, Tar)
            Eval, pred = Model_GRU(Image, Tar, sol)
            Eval = net_evaluation(Image, data)
            Fitn[i] = 1 / (Eval[7] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Image = Model_Yolov8(data, Tar)
        Eval, pred = Model_GRU(Image, Tar, sol)
        Eval = net_evaluation(Image, data)
        Fitn = 1 / (Eval[7] + Eval[6])
        return Fitn
