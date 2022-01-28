import torch
import json
import random
import numpy as np
from flask import Flask, request
from scripts.predict import predict_input
from utils.getTokenizer import getTokenizer

app = Flask(__name__)
tokenizer = getTokenizer("models/chat_DialoGPT_small_zh")
model = torch.load("checkpoint/modelColab.pt")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@app.route("/", methods=["POST"])
def messageBuild():
    data = request.get_json()
    msg = data.get("msg")
    result = predict_input(model, msg, tokenizer, "cpu")
    result = result.replace(tokenizer.pad_token, "")
    result = result.replace(tokenizer.bos_token, "")
    return json.dumps({"code":101, "message": result})


if __name__ == '__main__':
    config  = {
        "host": "0.0.0.0",
        "port": 9725
    }
    same_seeds(5211)
    app.run(**config)