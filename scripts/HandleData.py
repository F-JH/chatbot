import torch
from os.path import join
import numpy as np
import json
from tqdm.auto import tqdm


def run(tokenizer):
    print("开始处理")
    queFile = open("data/subtitle/que.txt", "r", encoding="utf-8")
    ansFile = open("data/subtitle/ans.txt", "r", encoding="utf-8")

    saveData = "data/subtitle"

    with open("data/subtitle/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    # writer = []

    queTokens = np.zeros((config["len"], config["queMax"]+1), dtype=np.int64) + tokenizer.pad_token_id
    ansTokens = np.zeros((config["len"], config["ansMax"]+1), dtype=np.int64) + tokenizer.pad_token_id
    labelTokens = np.zeros((config["len"], config["ansMax"]+1), dtype=np.int64) + tokenizer.pad_token_id

    for i in tqdm(range(config["len"])):
        que = queFile.readline()
        ans = ansFile.readline()
        if not que or not ans:
            break

        queToken = tokenizer.encode(que[:-1])
        ansToken = tokenizer.encode(ans[:-1])

        queToken = np.array(queToken, dtype=np.int64)
        ansToken = np.array(ansToken, dtype=np.int64)


        queTokens[i, 0:queToken.shape[0]] = queToken
        queTokens[i, queToken.shape[0]] = tokenizer.eos_token_id
        ansTokens[i, 0] = tokenizer.bos_token_id
        ansTokens[i, 1:ansToken.shape[0]+1] = ansToken
        labelTokens[i, 0:ansToken.shape[0]] = ansToken
        labelTokens[i, ansToken.shape[0]] = tokenizer.eos_token_id

    queFile.close()
    ansFile.close()
    print("finish")
    np.save(join(saveData, "que.npy"), queTokens)
    np.save(join(saveData, "ans.npy"), ansTokens)
    np.save(join(saveData, "label.npy"), labelTokens)
    print("save success")