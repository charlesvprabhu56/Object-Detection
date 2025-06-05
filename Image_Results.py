import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_dataset = 1


def Image_Results():
    for n in range(no_of_dataset):
        Orig = np.load('Image.npy', allow_pickle=True)
        segment = np.load('Segmentation_Image.npy', allow_pickle=True)
        cls = ["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorbike", "people", "table"]

        ind = [2, 839, 1393, 2010, 2423, 3118, 3852, 4545, 5002, 6166, 6320, 6889]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            fig, ax = plt.subplots(1, 2)
            plt.suptitle(cls[j], fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.subplot(1, 2, 2)
            plt.title('Seg')
            plt.imshow(seg)
            plt.show()


def Sample_Images():
    for n in range(no_of_dataset):
        Orig = np.load('Image.npy', allow_pickle=True)
        ind = [5, 750, 1390, 1980, 2415, 3059, 3846, 4489, 5029, 5752, 6285, 6790]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset " + str(n + 1))
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[5]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[6]])
        plt.show()



if __name__ == '__main__':
    Image_Results()
    Sample_Images()
