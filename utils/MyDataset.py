import json

import torch
import numpy as np
from os.path import join
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, queSentence, ansSentence, queMax, ansMax, tokenizer, n_head, mode="train"):
        self.queSentence = queSentence
        self.ansSentence = ansSentence
        self.queMax = queMax
        self.ansMax = ansMax
        self.tokenizer = tokenizer
        self.n_head = n_head
        self.mode = mode
        self.len = len(queSentence)
    def __len__(self):
        return self.len
    def __getitem__(self, ind):
        queSentence = self.queSentence[ind]
        ansSentence = self.ansSentence[ind]
        # 原始 token
        queToken = self.tokenizer.encode(queSentence) + [self.tokenizer.eos_token_id]
        ansToken = self.tokenizer.encode(ansSentence)
        label = ansToken + [self.tokenizer.eos_token_id]
        ansToken = [self.tokenizer.bos_token_id] + ansToken
        # 处理token和mask
        queToken = torch.LongTensor(queToken)
        ansToken = torch.LongTensor(ansToken)
        label = torch.LongTensor(label)
        queMask = torch.zeros(queToken.shape, dtype=torch.int64)
        ansMask = torch.zeros(ansToken.shape, dtype=torch.int64)
        quePad = self.queMax - queToken.shape[0]
        ansPad = self.ansMax - ansToken.shape[0]
        queToken = F.pad(queToken, [0, quePad], mode="constant", value=self.tokenizer.get_vocab().get("<|PAD|>"))
        ansToken = F.pad(ansToken, [0, ansPad], mode="constant", value=self.tokenizer.get_vocab().get("<|PAD|>"))
        label = F.pad(label, [0, ansPad], mode="constant", value=self.tokenizer.get_vocab().get("<|PAD|>"))
        queMask = F.pad(queMask, [0, quePad], mode="constant", value=1) # [m,]
        ansMask = F.pad(ansMask, [0, ansPad], mode="constant", value=1) # [m,]

        # 处理mask queMask、ansMask、queansMask
        queM = queMask.shape[0]
        ansM = ansMask.shape[0]
        queMask = queMask.unsqueeze(0)  # [1, m]
        queansMask = queMask.expand(ansM, queM)
        ansMask = ansMask.unsqueeze(0)
        ansMask = ansMask.expand(ansM, ansM)
        ansMask = ansMask.unsqueeze(0)
        ansMask = ansMask.expand(self.n_head, ansM, ansM)   # [n_head, ansM, ansM]

        queansMask = queansMask.unsqueeze(0)
        queansMask = queansMask.expand(self.n_head, ansM, queM) #[n_head, ansM, queM]

        queMask = queMask.expand(queM, queM)
        queMask = queMask.unsqueeze(0)
        queMask = queMask.expand(self.n_head, queM, queM)   # [n_head, queM, queM]

        if self.mode == "test":
            return queToken, queMask, ansToken, label, ansMask, queansMask, queSentence, ansSentence
        else:
            return queToken, queMask, ansToken, label, ansMask, queansMask

class HandleDataset(Dataset):
    def __init__(self, queTokens, ansTokens, labelTokens, n_head, mode="train"):
        # self.queTokens = torch.from_numpy(np.load(quePath))
        # self.ansTokens = torch.from_numpy(np.load(ansPath))
        # self.labelTokens = torch.from_numpy(np.load(labelPath))

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

def getDataset(dataPath, tokenizer, config, mode="normal"):
    if mode != "normal":
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
        trainDataset = HandleDataset(
            queTokens[0:num-(validNum+testNum)],
            ansTokens[0:num-(validNum+testNum)],
            labelTokens[0:num-(validNum+testNum)],
            config["transformerConfig"]["n_head"]
        )
        validDataset = HandleDataset(
            queTokens[num-(validNum+testNum):num-(testNum)],
            ansTokens[num-(validNum+testNum):num-(testNum)],
            labelTokens[num-(validNum+testNum):num-(testNum)],
            config["transformerConfig"]["n_head"]
        )
        testDataset = HandleDataset(
            queTokens[num-(testNum):],
            ansTokens[num-(testNum):],
            labelTokens[num-(testNum):],
            config["transformerConfig"]["n_head"]
        )
    else:
        n_head = config["transformerConfig"]["n_head"]
        queFile = open(join(dataPath, "que.txt"), "r", encoding="utf-8")
        ansFile = open(join(dataPath, "ans.txt"), "r", encoding="utf-8")

        with open(join(dataPath, "config.json"), "r") as c:
            configJson = json.load(c)
        queMax = configJson["queMax"]
        ansMax = configJson["ansMax"]
        queTrainSentence = []
        queValidSentence = []
        ansTrainSentence = []
        ansValidSentence = []
        queTestSentence = []
        ansTestSentence = []
        num = 1
        while True:
            que = queFile.readline()
            ans = ansFile.readline()
            if not que or not ans:
                break

            # 切分train和valid
            if num % config["dataset"]["validSplit"] == 0:
                queValidSentence.append(que[:-1])
                ansValidSentence.append(ans[:-1])
            elif num % config["dataset"]["testSplit"] == 0:
                queTestSentence.append(que[:-1])
                ansTestSentence.append(ans[:-1])
            else:
                queTrainSentence.append(que[:-1])
                ansTrainSentence.append(ans[:-1])
            num += 1
        trainDataset = MyDataset(queTrainSentence, ansTrainSentence, queMax+1, ansMax+2, tokenizer, n_head, "train")
        validDataset = MyDataset(queValidSentence, ansValidSentence, queMax+1, ansMax+2, tokenizer, n_head, "valid")
        testDataset = MyDataset(queTestSentence, ansTestSentence, queMax+1, ansMax+2, tokenizer, n_head, "test")
    return trainDataset, validDataset, testDataset