from numpy import inf
import torch
import random
import numpy as np
from utils import MyDataset, TransformerRuntimeMask
from torch.utils.data import DataLoader
from config import config
from torch import nn,optim
from utils.getTokenizer import getTokenizer
from scripts.SaveLoad import loadCheckpoint, loadWeight
from utils.schedule import get_cosine_schedule_with_warmup

# from scripts.train import valid, predict, train
from scripts.TrainRuntimeMask import trainRuntimeMask, predict, predict_input

from os.path import exists

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(5211)

def main():
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)

    trainDataset, validDataset, testDataset = MyDataset.getDataset(config.dataPath, config.trainConfig)
    # model = transformer.Transformer(len(tokenizer.get_vocab().keys()), **config.trainConfig["transformerConfig"])
    model = TransformerRuntimeMask.Transformer(len(tokenizer.get_vocab().keys()), tokenizer.pad_token_id, **config.trainConfig["transformerConfig"])
    if config.trainConfig["ignore_pad_loss"]:
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.trainConfig["optim"])(model.parameters(), lr=config.trainConfig["lr"], **config.trainConfig["params"])

    trainDataLoader = DataLoader(trainDataset, batch_size=config.trainConfig["batch_size"], shuffle=True, pin_memory=True, num_workers=config.trainConfig["dataset"]["num_workers"])
    validDataLoader = DataLoader(validDataset, batch_size=config.trainConfig["batch_size"], shuffle=True, pin_memory=True, num_workers=config.trainConfig["dataset"]["num_workers"])
    print("size of train:", len(trainDataset))
    print("size of valid:", len(validDataset))
    print("size of test:", len(testDataset))

    total_steps = int(-(len(trainDataset) // -config.trainConfig["batch_size"]) * config.trainConfig["n_epochs"])
    warmup_steps = total_steps // 10
    print("warmup_steps:", warmup_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    epoch = 0
    batch_n = 0
    bestLoss = inf
    if exists(config.trainConfig["savepoint"]):
        model, optimizer, bestLoss, epoch, batch_n, scheduler = loadCheckpoint(model, optimizer, scheduler, config.trainConfig["savepoint"], "cuda")
    trainRuntimeMask(model, epoch, batch_n, bestLoss, trainDataLoader, validDataLoader, testDataset, optimizer, criterion, scheduler, tokenizer, config.trainConfig)

    # que, ans = predict(model, testDataset, tokenizer, 3)
    # print("test: [que]{} | [ans]{}".format("".join(que), "".join(ans)))

def test():
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)
    model = TransformerRuntimeMask.Transformer(len(tokenizer.get_vocab().keys()), tokenizer.pad_token_id, **config.trainConfig["transformerConfig"])
    if exists(config.trainConfig["savepoint"]):
        # optimizer = getattr(optim, config.trainConfig["optim"])(model.parameters(), lr=config.trainConfig["lr"], **config.trainConfig["params"])
        # scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 1000)
        # model, optimizer, bestLoss, epoch, batch_n, scheduler = loadCheckpoint(model, optimizer, scheduler, config.trainConfig["savepoint"], "cpu")
        model = loadWeight(model, config.trainConfig["savepoint"], "cpu")
    while True:
        msg = input(">>>")
        if(msg == "exit"):
            break
        result = predict_input(model, msg, tokenizer)
        result = result.replace(tokenizer.pad_token, "")
        result = result.replace(tokenizer.bos_token, "")
        print(result)

if __name__ == '__main__':
    same_seeds(5211)
    main()
    # test()