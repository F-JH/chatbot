import json

import torch
import numpy as np
from os.path import join
import torch.nn.functional as F
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, queTokens, ansTokens, labelTokens, n_head, mode="train"):
        self.queTokens = queTokens
        self.ansTokens = ansTokens
        self.labelTokens = labelTokens

        self.len = self.queTokens.shape[0]
        self.n_head = n_head
        self.mode = mode
    def __getitem__(self, item):
        return self.queTokens[item], self.ansTokens[item], self.labelTokens[item]
    def __len__(self):
        return self.len

def getDataset(dataPath, config):
    print("load data from npy...")
    quePath = join(dataPath, "que.npy")
    ansPath = join(dataPath, "ans.npy")
    labelPath = join(dataPath, "label.npy")

    queTokens = torch.from_numpy(np.load(quePath))
    ansTokens = torch.from_numpy(np.load(ansPath))
    labelTokens = torch.from_numpy(np.load(labelPath))
    print("success")
    num = queTokens.shape[0]
    validNum = config["dataset"]["validNum"]
    testNum = config["dataset"]["testNum"]
    trainDataset = ChatbotDataset(
        queTokens[0:num-(validNum+testNum)],
        ansTokens[0:num-(validNum+testNum)],
        labelTokens[0:num-(validNum+testNum)],
        config["transformerConfig"]["n_head"]
    )
    validDataset = ChatbotDataset(
        queTokens[num-(validNum+testNum):num-(testNum)],
        ansTokens[num-(validNum+testNum):num-(testNum)],
        labelTokens[num-(validNum+testNum):num-(testNum)],
        config["transformerConfig"]["n_head"]
    )
    testDataset = ChatbotDataset(
        queTokens[num-(testNum):],
        ansTokens[num-(testNum):],
        labelTokens[num-(testNum):],
        config["transformerConfig"]["n_head"]
    )
    return trainDataset, validDataset, testDataset