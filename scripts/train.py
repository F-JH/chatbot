import torch
import numpy as np
from tqdm.auto import tqdm
from os.path import join
import json, wandb
from config.settings import dataPath
from scripts.SaveLoad import saveCheckpoint

def get_attn_mask(len_q, n_head, seq_k, PADMASK):
    batch_size, len_k = seq_k.shape
    padMask = seq_k.data.eq(PADMASK).unsqueeze(1)   # [batch_size, 1, len_k]
    padMask = padMask.expand(batch_size, len_q, len_k)
    padMask = padMask.unsqueeze(1)
    return padMask.expand(batch_size, n_head, len_q, len_k)

def valid(model, criterion, validData):
    model.eval()
    validLoss = 0
    n = 0
    for batch in tqdm(validData):
        n += 1
        batch = [i.to("cuda") for i in batch]
        with torch.no_grad():
            output = model(batch[0], batch[1], batch[2], batch[4], batch[5])
            loss = criterion(output, batch[3].view(-1))
            validLoss += loss.item()
    validLoss /= n
    return validLoss

def predict(model, testData, tokenizer, n_head):
    with open(join(dataPath, "config.json"), "r") as c:
        config = json.load(c)
    model.eval()
    next_symbol = tokenizer.bos_token_id
    data = testData[int(np.random.rand() * len(testData))]
    # data = [i.to("cuda") for i in data]
    queToken, queMask, ansToken, _, _, _, queSentence, ansSentence = data
    queToken = queToken.to("cuda")
    queMask = queMask.to("cuda")
    queToken = queToken.unsqueeze(0)
    queMask = queMask.unsqueeze(0)

    with torch.no_grad():
        encOutput = model.encoder(queToken, queMask)
        decInput = torch.zeros(1, config["ansMax"], dtype=torch.long) + tokenizer.pad_token_id
        decInput = decInput.to("cuda")
        decInput[0, 0] = next_symbol
        for i in range(config["ansMax"]):
            # 制作mask: decMask, encdecMask
            decMask = get_attn_mask(config["ansMax"], n_head, decInput, tokenizer.pad_token_id)
            encdecMask = get_attn_mask(config["ansMax"], n_head, queToken, tokenizer.pad_token_id)
            # 开始解码
            decOutput = model.decoder(decInput, encOutput, decMask, encdecMask)
            decOutput = model.generator(decOutput)  # [1, ansMax, vocab_size]
            decOutput = decOutput.squeeze(0)
            next_symbol = decOutput[i, :].argmax()
            if next_symbol == tokenizer.eos_token_id:
                break
            if i != config["ansMax"] - 1:
                decInput[0, i+1] = next_symbol
    return tokenizer.decode(queToken.squeeze(0)), tokenizer.decode(decInput.squeeze(0))


def train(model, epoch, bestLoss, trainData, validData, testData, optimizer, criterion, scheduler, tokenizer, config):
    if config["log"]:
        wandb.init()
    while epoch < config["n_epochs"]:
        model.train()
        totalLoss = 0
        n = 0
        for data in tqdm(trainData):
            n += 1
            optimizer.zero_grad()
            data = [i.to("cuda") for i in data]
            output = model(data[0], data[1], data[2], data[4], data[5])
            loss = criterion(output, data[3].view(-1))
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 等不了一个epoch了，太多了
            if n % 4000 == 0:
                # print("start valid...")
                totalLoss /= n
                validLoss = valid(model, criterion, validData)
                if config["log"]:
                    log = {
                        "train loss": totalLoss,
                        "valid loss": validLoss
                    }
                    wandb.log(log)
                msg = "epoch : {}| Train Loss: {}| Valid Loss: {}".format(epoch, totalLoss, validLoss)
                print(msg)
                que, ans = predict(model, testData, tokenizer, config["transformerConfig"]["n_head"])
                print("test: [que]{} | [ans]{}".format("".join(que), "".join(ans)))
                if validLoss < bestLoss:
                    bestLoss = validLoss
                    print("save model................................................")
                    saveCheckpoint(model, optimizer,scheduler, bestLoss, epoch, config["savepoint"])
                totalLoss = 0
                n = 0

        # totalLoss /= n
        # validLoss = valid(model, criterion, validData)
        # if config["log"]:
        #     log = {
        #         "train loss": totalLoss,
        #         "valid loss": validLoss
        #     }
        #     wandb.log(log)
        # msg = "epoch: {}| Train Loss: {}| Valid Loss: {}".format(epoch, totalLoss, validLoss)
        # print(msg)
        # test = predict(model, testData, tokenizer)
        # print("test:", test)
        # if validLoss < bestLoss:
        #     bestLoss = validLoss
        #     print("save model................................................")
        #     saveCheckpoint(model, optimizer,scheduler, bestLoss, epoch, config["savepoint"])