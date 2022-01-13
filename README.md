# chatbot

用Transformer训练的一个chatbot，业余玩家，随便玩玩
数据集：
subtitle（电视剧对白语料，单轮）https://github.com/fateleak/dgk_lost_conv  

#### 快速开始训练

```
1.预处理数据：python PreHandleData.py
--处理完后，在data/subtitle下应出现三个npy文件：que.npy、ans.npy、label.npy，分别是encInput、decInput、和label的使用tokenizer.encode过的数据
2.开始训练：python main.py
```

#### 配置

```
所有配置都在config/config.py下
trainConfig = {
    "n_epochs": 1,
    "log": True,
    "lr": 5e-6,
    "batch_size": 64,
    "rest_batch": 1000,
    "optim": "Adam",
    "dataset": {
        # 这个计算参考 utils/MyDataset.py 87行，不想使用百分比来控制valid和test集大小，各位不喜欢的话可以自行修改代码(现在已经不用这个方法了)
        "validSplit": 430,
        "testSplit": 500,
        # 验证集和测试及大小，num_workers: Dataloader的num_workers参数，建议各位视自己机器的情况而定，我是6核
        "validNum": 10000,
        "testNum": 5000,
        "num_workers": 6
    },
    "params": {
        "betas": (0.9, 0.99),
        "eps": 1e-8,
    },
    "transformerConfig":{
        "d_model": 60,
        "n_head": 3,
        "num_of_layer": 8
    },
    "savepoint": "checkpoint/checkpoint.pth"
}
```
