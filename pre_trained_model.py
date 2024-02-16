import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
import torchvision
import torchvision.datasets as datasets
from handcrafted.app.extractor.hog_extractor import HOGExtractor
from handcrafted.app.dataset.dataset import Dataset
from pre_trained_model.model.C3DModel import C3DModel
import gluoncv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    dataset = Dataset("data/WLASL_v0.3.json")

