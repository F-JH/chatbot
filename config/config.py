preDataFile = "chinese_chatbot_corpus/clean_chat_corpus/xiaohuangji.tsv"
dataPath = "data/xiaohuangji"
sentenceMaxLen = 50     # 预处理中，每条对话每个sentence的最大词长度（长度是指被tokenizer encode过的token长度）
trainConfig = {
    "n_epochs": 10,
    "log": True,
    "lr": 3e-4,
    "batch_size": 32,
    "rest_batch": 2000,
    "optim": "Adam",
    "ignore_pad_loss": True,    # 计算loss时是否忽略pad
    "dataset": {
        # 验证集和测试及大小，num_workers: Dataloader的num_workers参数，建议各位视自己机器的情况而定，我是6核
        "validNum": 20000,
        "testNum": 5000,
        "num_workers": 6
    },
    "params": {
        "betas": (0.9, 0.99),
        "eps": 1e-8,
    },
    "transformerConfig":{
        # d_model 必须能被 n_head 整除，且为符合positionEmbedding，d_model要是偶数
        "d_model": 80,
        "n_head": 5,
        "num_of_layer": 6,
        "device": "cpu"
    },
    "savepoint": "checkpoint/checkpoint.pth"
}