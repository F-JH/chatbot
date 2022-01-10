import json

import torch
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

def getDataset(dataPath, tokenizer, n_head):
    queFile = open(join(dataPath, "que.txt"), "r", encoding="utf-8")
    ansFile = open(join(dataPath, "ans.txt"), "r", encoding="utf-8")

    with open(join(dataPath, "config.json"), "r") as c:
        config = json.load(c)
    queMax = config["queMax"]
    ansMax = config["ansMax"]
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
        if num % 430 == 0:
            queValidSentence.append(que[:-1])
            ansValidSentence.append(ans[:-1])
        elif num % 500 == 0:
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