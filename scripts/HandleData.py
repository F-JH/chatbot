import torch
from os.path import join
import numpy as np
import json
import re
from config import config
from tqdm.auto import tqdm

def symbols(sentence):
    p = "[` ~!@#$%^&*()_\-+=<>?:\"{}|,.\/;'\\[\]·~！@#￥%……&*（）——\-+={}|《》？：“”【】、；‘'，。、]+"
    match = re.match(p, sentence)
    if match is None:
        return False
    if match.span()[1] == len(sentence):
        return True
    return False

def run(tokenizer, dataFile, savePath):
    print("开始处理")
    maxLen = config.sentenceMaxLen
    queTokens = []
    ansTokens = []
    labelTokens = []
    with open(dataFile, "r", encoding="utf-8") as f:
        n = 0
        while True:
            print("\r[{}]".format(n), end="")
            n += 1
            sentences = f.readline()
            if not sentences:
                break
            if config.is_replace_space:
                sentences = sentences.replace(" ", "")
            sentences = sentences.replace("\n", "")
            sentences = sentences.split("\t")
            if symbols(sentences[0]) or symbols(sentences[1]):
                # 全是特殊符号，没有什么意义
                continue
            que = tokenizer.encode(sentences[0])
            ans = tokenizer.encode(sentences[1])
            if len(que) > maxLen or len(ans)+1 > maxLen:
                continue
            que = que + [tokenizer.pad_token_id] * (maxLen - len(que))
            label = ans + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (maxLen - len(ans) - 1)
            ans = [tokenizer.bos_token_id] + ans + [tokenizer.pad_token_id] * (maxLen - len(ans) - 1)

            queTokens.append(que)
            ansTokens.append(ans)
            labelTokens.append(label)
    queTokens = np.array(queTokens, dtype=np.int64)
    ansTokens = np.array(ansTokens, dtype=np.int64)
    labelTokens = np.array(labelTokens, dtype=np.int64)
    print("size of data:", len(queTokens))
    print("save data...")
    np.save(join(savePath, "que.npy"), queTokens)
    np.save(join(savePath, "ans.npy"), ansTokens)
    np.save(join(savePath, "label.npy"), labelTokens)

# def run(tokenizer):
#     print("开始处理")
#     queFile = open("data/subtitle/que.txt", "r", encoding="utf-8")
#     ansFile = open("data/subtitle/ans.txt", "r", encoding="utf-8")
#
#     saveData = "data/subtitle"
#
#     with open("data/subtitle/config.json", "r", encoding="utf-8") as f:
#         config = json.load(f)
#     # writer = []
#
#     queTokens = np.zeros((config["len"], config["queMax"]), dtype=np.int64) + tokenizer.pad_token_id
#     ansTokens = np.zeros((config["len"], config["ansMax"]+1), dtype=np.int64) + tokenizer.pad_token_id
#     labelTokens = np.zeros((config["len"], config["ansMax"]+1), dtype=np.int64) + tokenizer.pad_token_id
#
#     for i in tqdm(range(config["len"])):
#         que = queFile.readline()
#         ans = ansFile.readline()
#         if not que or not ans:
#             break
#
#         queToken = tokenizer.encode(que[:-1])
#         ansToken = tokenizer.encode(ans[:-1])
#
#         queToken = np.array(queToken, dtype=np.int64)
#         ansToken = np.array(ansToken, dtype=np.int64)
#
#
#         queTokens[i, 0:queToken.shape[0]] = queToken
#         # queTokens[i, queToken.shape[0]] = tokenizer.eos_token_id
#         ansTokens[i, 0] = tokenizer.bos_token_id
#         ansTokens[i, 1:ansToken.shape[0]+1] = ansToken
#         labelTokens[i, 0:ansToken.shape[0]] = ansToken
#         labelTokens[i, ansToken.shape[0]] = tokenizer.eos_token_id
#
#     queFile.close()
#     ansFile.close()
#     print("finish")
#     np.save(join(saveData, "que.npy"), queTokens)
#     np.save(join(saveData, "ans.npy"), ansTokens)
#     np.save(join(saveData, "label.npy"), labelTokens)
#     print("save success")