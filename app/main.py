import cv2
import matplotlib.pyplot as plt
from utilities.preprocess import preprocess

if __name__ == '__main__':
    fg = preprocess('../data/videos/00336.mp4')
    plt.imshow(fg)
    plt.axis('off')
    plt.show()
    # cv2.imshow('test', fg)

