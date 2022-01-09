import json

import torch
from os.path import join
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, queSentence, ansSentence, queMax, ansMax, tokenizer):
        self.queSentence = queSentence
        self.ansSentence = ansSentence
        self.queMax = queMax
        self.ansMax = ansMax
        self.tokenizer = tokenizer
        self.len = len(queSentence)
    def __len__(self):
        return self.len
    def __getitem__(self, ind):
        queSentence = self.queSentence[ind]
        ansSentence = self.ansSentence[ind]
        # 原始 token
        queToken = self.tokenizer.encode(queSentence) + [self.tokenizer.eos_token_id]
        ansToken = [self.tokenizer.bos_token_id] + self.tokenizer.encode(ansSentence) + [self.tokenizer.eos_token_id]
        # 处理token和mask
        queToken = torch.LongTensor(queToken)
        ansToken = torch.LongTensor(ansToken)
        queMask = torch.ones(queToken.shape, dtype=torch.int64)
        ansMask = torch.ones(ansToken.shape, dtype=torch.int64)
        quePad = self.queMax - queToken.shape[0]
        ansPad = self.ansMax - ansToken.shape[0]
        queToken = F.pad(queToken, [0, quePad], mode="constant", value=self.tokenizer.get_vocab().get("<|PAD|>"))
        ansToken = F.pad(ansToken, [0, ansPad], mode="constant", value=self.tokenizer.get_vocab().get("<|PAD|>"))
        queMask = F.pad(queMask, [0, quePad])
        ansMask = F.pad(ansMask, [0, ansPad])

        return queToken, queMask, ansToken, ansMask

def getDataset(dataPath, tokenizer):
    queFile = open(join(dataPath, "que.txt"), "r", encoding="utf-8")
    ansFile = open(join(dataPath, "ans.txt"), "r", encoding="utf-8")

    with open(join(dataPath, "config.json"), "r") as c:
        config = json.load(c)
    queMax = config["queMax"]
    ansMax = config["ansMax"]
    queSentence = []
    ansSentence = []
    while True:
        que = queFile.readline()
        ans = ansFile.readline()

        if not que or not ans:
            break
        queSentence.append(que[:-1])
        ansSentence.append(ans[:-1])
    return MyDataset(queSentence, ansSentence, queMax+1, ansMax+2, tokenizer)