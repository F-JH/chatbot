import torch
import numpy as np
from tqdm.auto import tqdm
from os.path import join
import json, wandb
from config.config import dataPath, sentenceMaxLen
from scripts.SaveLoad import saveCheckpoint
import torch.nn.functional as F

def valid(model, criterion, validData):
    model.eval()
    validLoss = 0
    n = 0
    for batch in tqdm(validData):
        n += 1
        batch = [i.to("cuda") for i in batch]
        with torch.no_grad():
            output = model(batch[0], batch[1])
            loss = criterion(output, batch[2].view(-1))
            validLoss += loss.item()
    validLoss /= n
    return validLoss

def predict(model, testData, tokenizer, n_head):
    # with open(join(dataPath, "config.json"), "r") as c:
    #     config = json.load(c)
    model.eval()
    next_symbol = tokenizer.bos_token_id
    data = testData[int(np.random.rand() * len(testData))]
    # data = [i.to("cuda") for i in data]
    queToken, ansToken, labelToken = data
    queToken = queToken.unsqueeze(0)
    queToken = queToken.to("cuda")
    n = 0
    with torch.no_grad():
        encOutput = model.encoder(queToken)
        decInput = torch.zeros(1, ansToken.shape[0], dtype=torch.long) + tokenizer.pad_token_id
        decInput = decInput.to("cuda")
        decInput[0, 0] = next_symbol
        for i in range(ansToken.shape[0]):
            n += 1
            # 开始解码
            decOutput = model.decoder(queToken, decInput, encOutput)
            decOutput = model.generator(decOutput)  # [1, ansMax, vocab_size]
            decOutput = decOutput.squeeze(0)
            next_symbol = decOutput[i, :].argmax()
            if next_symbol == tokenizer.eos_token_id:
                break
            if i != ansToken.shape[0] - 1:
                decInput[0, i+1] = next_symbol
    return tokenizer.decode(queToken.squeeze(0)), tokenizer.decode(decInput.squeeze(0)[0:n-1]), tokenizer.decode(labelToken), n

def predict_input(model, msg, tokenizer):
    model.eval()
    next_symbol = tokenizer.bos_token_id
    # data = [i.to("cuda") for i in data]
    queToken = torch.LongTensor(tokenizer.encode(msg))
    queToken = F.pad(queToken, [0, sentenceMaxLen - queToken.shape[0]], mode="constant", value=tokenizer.pad_token_id)
    queToken = queToken.unsqueeze(0)
    queToken = queToken.to("cuda")
    n = 0
    with torch.no_grad():
        encOutput = model.encoder(queToken)
        decInput = torch.zeros(1, sentenceMaxLen, dtype=torch.long) + tokenizer.pad_token_id
        decInput = decInput.to("cuda")
        decInput[0, 0] = next_symbol
        for i in range(sentenceMaxLen):
            n += 1
            # 开始解码
            decOutput = model.decoder(queToken, decInput, encOutput)
            decOutput = model.generator(decOutput)  # [1, ansMax, vocab_size]
            decOutput = decOutput.squeeze(0)
            next_symbol = decOutput[i, :].argmax()
            if next_symbol == tokenizer.eos_token_id:
                break
            if i != sentenceMaxLen-1:
                decInput[0, i+1] = next_symbol
    return tokenizer.decode(decInput.squeeze(0))

def trainRuntimeMask(model, epoch, batch_n, bestLoss, trainData, validData, testData, optimizer, criterion, scheduler, tokenizer, config):
    use_load_batch_n = True
    if config["log"]:
        wandb.init()
    count = 0
    while epoch < config["n_epochs"]:
        model.train()
        totalLoss = 0
        for data in tqdm(trainData):
            if use_load_batch_n and count < batch_n:
                count += 1
                continue
            else:
                use_load_batch_n = False
            batch_n += 1
            optimizer.zero_grad()
            data = [i.to("cuda") for i in data]
            output = model(data[0], data[1])
            loss = criterion(output, data[2].view(-1))
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 等不了一个epoch了，太多了
            if batch_n % config["rest_batch"] == 0:
                # print("start valid...")
                totalLoss /= config["rest_batch"]
                validLoss = valid(model, criterion, validData)
                if config["log"]:
                    log = {
                        "train loss": totalLoss,
                        "valid loss": validLoss
                    }
                    wandb.log(log)
                msg = "epoch : {}| Train Loss: {}| Valid Loss: {}".format(epoch, totalLoss, validLoss)
                print(msg)
                que, ans, label, n = predict(model, testData, tokenizer, config["transformerConfig"]["n_head"])
                que = "".join(que).replace(tokenizer.pad_token, "")
                ans = "".join(ans).replace(tokenizer.pad_token, "")
                label = "".join(label).replace(tokenizer.pad_token, "")
                print("test:{}\n[que]{}\n[ans]{}\n[label]{}".format(n, que, ans, label))
                if validLoss < bestLoss:
                    bestLoss = validLoss
                    print("save model................................................")
                    saveCheckpoint(model, optimizer,scheduler, bestLoss, epoch, batch_n, config["savepoint"])
                totalLoss = 0
        batch_n = 0