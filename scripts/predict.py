import torch
import numpy as np
from config.config import sentenceMaxLen
import torch.nn.functional as F

def predict(model, testData, tokenizer, n_head):
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

def predict_input(model, msg, tokenizer, device):
    model.eval()
    next_symbol = tokenizer.bos_token_id
    # data = [i.to("cuda") for i in data]
    queToken = torch.LongTensor(tokenizer.encode(msg))
    queToken = F.pad(queToken, [0, sentenceMaxLen - queToken.shape[0]], mode="constant", value=tokenizer.pad_token_id)
    queToken = queToken.unsqueeze(0)
    queToken = queToken.to(device)
    n = 0
    with torch.no_grad():
        encOutput = model.encoder(queToken)
        decInput = torch.zeros(1, sentenceMaxLen, dtype=torch.long) + tokenizer.pad_token_id
        decInput = decInput.to(device)
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