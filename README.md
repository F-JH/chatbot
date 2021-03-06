# chatbot

用Transformer训练的一个chatbot，业余玩家，随便玩玩
数据集：[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)

#### 下载数据集到项目中

* git clone 上面那个项目到根目录，然后按照他的文档下载相应的数据集，然后会生成目录 chinese_chatbot_corpus/clean_chat_corpus，处理过的语料都在这里面，到config/config.py设置好preDataFile，指定自己要的语料（感觉小黄鸡的效果不错，可能以后会自己收集一些语料，但不会放出来）

#### 快速开始训练

```
1.预处理数据：python PreHandleData.py
--处理完后，在config/config.py:dataPath这个配置的目录下应出现三个npy文件：que.npy、ans.npy、label.npy，分别是encInput、decInput、和label的使用tokenizer.encode过的数据
2.开始训练：python main.py
```

#### 配置

```
所有配置都在config/config.py下
trainConfig = {
    "n_epochs": 10,
    "log": True,         # 是否连到wandb记录训练过程，需要翻墙，各位翻不了的就设为Flase吧
    "lr": 3e-4,
    "batch_size": 32,
    "rest_batch": 2000,
    "optim": "Adam",
    "ignore_pad_loss": True,    # 计算loss时是否忽略pad
    "dataset": {
        # 这个计算参考 utils/MyDataset.py 87行，不想使用百分比来控制valid和test集大小，各位不喜欢的话可以自行修改代码(现在已经不用这个方法了)
        "validSplit": 430,
        "testSplit": 500,
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
        # d_model 必须能被 n_head 整除，且为了符合positionEmbedding，d_model要是偶数
        "d_model": 80,
        "n_head": 5,
        "num_of_layer": 6
    },
    "savepoint": "checkpoint/checkpoint.pth"
}
```
