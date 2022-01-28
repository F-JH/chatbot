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
            for qkey in config.preReplace["que"]:
                sentences[0] = sentences[0].replace(qkey, config.preReplace["que"][qkey])
            for akey in config.preReplace["ans"]:
                sentences[1] = sentences[1].replace(akey, config.preReplace["ans"][akey])
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
