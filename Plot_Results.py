import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSO-R3DT-YAGRU', 'DMO-R3DT-YAGRU', 'DBO-R3DT-YAGRU', 'DOA-R3DT-YAGRU', 'MRV-DOA-R3DT-YAGRU']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[:, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[2])
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='RSO-R3DT-YAGRU')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='DMO-R3DT-YAGRU')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='DBO-R3DT-YAGRU')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='DOA-R3DT-YAGRU')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='MRV-DOA-R3DT-YAGRU')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def plot_Result():
    eval = np.load('Eval_Hidden.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'MAP']
    Graph_Terms = [0, 1, 2, 3]
    Algorithm = ['TERMS', 'RSO-R3DT-YAGRU', 'DMO-R3DT-YAGRU', 'DBO-R3DT-YAGRU', 'DOA-R3DT-YAGRU', 'MRV-DOA-R3DT-YAGRU']
    Classifier = ['TERMS', 'DenseNet', 'MobileNetV2', 'LSTM', 'R3DT-YGRU', 'MRV-DOA-R3DT-YAGRU']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    epoch = ['100', '200', '300', '400', '500']
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[2]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[2]):
                for i in range(eval.shape[0]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

        fig = plt.figure()
        fig.canvas.manager.set_window_title(' Epoch for Method Comparison')
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        X = np.arange(len(epoch))
        ax.bar(X + 0.00, Graph[:, 5], color='royalblue', width=0.15, label="DenseNet")
        ax.bar(X + 0.15, Graph[:, 6], color='violet', width=0.15, label="MobileNetV2")
        ax.bar(X + 0.30, Graph[:, 7], color='palegreen', width=0.15, label="LSTM")
        ax.bar(X + 0.45, Graph[:, 8], color='crimson', width=0.15, label="R3DT-YGRU")
        ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label="MRV-DOA-R3DT-YAGRU")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                   ncol=3, fancybox=True, shadow=True)
        plt.xticks(X + 0.20, ('100', '200', '300', '400', '500'))
        plt.xlabel('Epochs')
        plt.ylabel(Terms[Graph_Terms[j]])
        path1 = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_seg_results():
    Eval_all = np.load('Eval_all1.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'ROA', 'GTO', 'AOA', 'FDA', 'PROPOSED']
    Methods = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'ARMNet', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1], value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1]):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            fig.canvas.manager.set_window_title(str(Terms[i - 4]))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='#9F79EE', width=0.15, label="RSO-R3DT-YAGRU")
            ax.bar(X + 0.15, stats[i, 1, :], color='#FF34B3', width=0.15, label="DMO-R3DT-YAGRU")
            ax.bar(X + 0.30, stats[i, 2, :], color='#8DB6CD', width=0.15, label="DBO-R3DT-YAGRU")
            ax.bar(X + 0.45, stats[i, 3, :], color='#EE9572', width=0.15, label="DOA-R3DT-YAGRU")
            ax.bar(X + 0.60, stats[i, 4, :], color='k', width=0.15, label="MRV-DOA-R3DT-YAGRU")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path1 = "./Results/%s_alg.png" % (Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            fig.canvas.manager.set_window_title(str(Terms[i - 4]))
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='r', width=0.15, label="Unet")
            ax.bar(X + 0.15, stats[i, 6, :], color='g', width=0.15, label="Unet++")
            ax.bar(X + 0.30, stats[i, 7, :], color='b', width=0.15, label="ResUnet")
            ax.bar(X + 0.45, stats[i, 8, :], color='m', width=0.15, label="R3DT-YGRU")
            ax.bar(X + 0.60, stats[i, 4, :], color='k', width=0.15, label="MRV-DOA-R3DT-YAGRU")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path = "./Results/%s_met.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_Result()
    plot_seg_results()
