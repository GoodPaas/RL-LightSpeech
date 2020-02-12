# RL-LightSpeech
使用Reinforcement Learning去实现NAT without a AT teacher model。

## 概要
FastSpeech使用了一个teacher model来提供alignment，这里的teacher model是Transformer-TTS，在我复现的版本里面使用的是Tacotron2，最近读到的论文DurIAN使用的是Forced Alignment工具，总之它们都使用了一个teacher model来提供alignment信息，训练并不是完全的端到端化的，能不能不用一个teacher model来实现NAT model？前几天review了一下NAT NMT的paper，还没有找到可以不需要teacher model的NAT NMT，如果有读者知道也可以在评论区交流一下。我尝试了一下使用强化学习的方法来让模型找到alignment，并没有成功，下面是我的主要思路。

## 设计
### reward的定义
我使用了真实mel声谱图和预测mel声谱图之间的MSELoss的倒数作为reward，具体的代码定义如下：
```Python
rewards = list()
for batch_ind in range(mel.size(0)):
    len_cut = length_target[batch_ind]
    mel_target_cut = mel_target[batch_ind][:len_cut]
    mel_pred_cut = mel[batch_ind][:len_cut]
    mat = 1.0 / (torch.pow(mel_pred_cut-mel_target_cut, 2) + 1.0)
    rewards.append(torch.sum(torch.sum(mat, -1), -1).item())

rewards = torch.Tensor(rewards).to("cuda").reshape((-1, 1))
rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # 归一化
```

### Actor和Environment的定义
我将Encoder和Length Regulator模块视为Actor，Decoder和Mel Target视为Environment

### Policy Gradient
loss：
```Python
pg_loss = torch.sum(torch.sum(torch.sum(torch.mul(rewards, history).mul(-1), -1), -1), -1) / rewards.size(0)
```

history（layers.py 第130行）：
```Python
m = Categorical(duration_predictor_P[i])
history.append(m.log_prob(m.sample()))
```

### 注意
- 这里不存在传统强化学习中episode的概念，只有一轮次的预测，相当于游戏只有一次动作
- 有两个loss，一个是正常训练的loss，一个是强化学习的loss，分别做反向传播，第一个loss修改的是整个模型的参数，第二个loss修改的是actor的参数

## 结果
训练了60000步，loss迟迟下降不了，推测原因：
1. 搜索空间太大；
2. 预测出来的mel声谱图和groud truth没有对齐，计算出来的loss不具有实际意义
