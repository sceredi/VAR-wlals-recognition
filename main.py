import matplotlib.pyplot as plt
from app.dataset.dataset import Dataset
from app.utilities.preprocess import preprocess

if __name__ == '__main__':
    fg = preprocess('data/videos/00336.mp4')
    plt.imshow(fg)
    plt.axis('off')
    plt.show()
    dataset = Dataset('data/WLASL_v0.3.json')
    print(dataset.videos)

