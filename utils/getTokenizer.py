from transformers import AutoTokenizer

def getTokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 增加特殊token
    tokens = {
        "bos_token": "<|BOS|>",
        "eos_token": "<|EOS|>",
        "pad_token": "<|PAD|>",
    }
    tokenizer.add_special_tokens(tokens)
    # 增加query和answer角色的token
    tokenizer.add_tokens(["<|QUE|>", "<|ANS|>"])
    return tokenizer