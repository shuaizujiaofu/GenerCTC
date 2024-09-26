import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange
from tqdm import tqdm
import pickle
import utils
import os
import json

from model.model_mlp import Mlp
from scripts.preprocessing.preprocess_banking import load_data_banking
from model.loss_cosent import CoSENTLoss
from model.loss_tangent import MultiClassTgLoss


def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    logits = net(X)
    l = loss(logits, y)
    l.sum().backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(logits, y)

    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    num_batches = len(train_iter)
    best_acc = 0

    for epoch in trange(num_epochs, desc="Epoch", disable=False):
        # Sum of training loss, sum of training spearman, no. of examples, no. of predictions
        metric = d2l.Accumulator(4)
        train_iterators = iter(train_iter)
        for i in trange(len(train_iter), desc="Iteration", smoothing=0.05, disable=False):
            data = next(train_iterators)
            features, labels = data
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print("train loss =", metric[0] / metric[2])
                print("train acc =", metric[1] / metric[3])
                metric.reset()
        test_acc = utils.evaluate_accuracy_gpu(net, test_iter)
        print("test acc =", test_acc)
        if test_acc - best_acc > 0.0001:
            best_acc = test_acc
    print(f'best acc {best_acc:.4f}')

    return best_acc


if __name__ == '__main__':
    encoder_config_path = "./../../config/bert_mini.json"
    with open(os.path.normpath(encoder_config_path), 'r') as config_file:
        encoder_config = json.load(config_file)
    dataset_config_path = "./../../config/mlp/banking.json"
    with open(os.path.normpath(dataset_config_path), 'r') as config_file:
        dataset_config = json.load(config_file)
    devices = d2l.try_all_gpus()

    # 设置超参数
    batch_size, num_steps = 64, 75
    pool = "cls_before_pooler"

    test_acc_list = []
    for i in range(5):
        # 读取数据集
        train_iter, test_iter = load_data_banking(batch_size, encoder_config, dataset_config, num_steps, False)

        # 创建模型
        net = Mlp(encoder_config, dataset_config["num_labels"], pool=pool)

        # 加载预训练模型
        # pretrained_file = '../model/ssc/banking/2-1-5.pth.tar'
        # checkpoint = torch.load(pretrained_file, map_location=torch.device('cpu'))
        # net.load_state_dict(checkpoint["model"], strict=False)

        # 定义参数
        lr, num_epochs = 3e-5, 20
        param_optimizer = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # warmup_steps = math.ceil(len(train_iter) * num_epochs * 0.1)  # 10% of train data for warm-up
        # steps_per_epoch = min([len(train_iter) for train_iter in train_iter])

        # 设置损失
        # loss = nn.CrossEntropyLoss(reduction="none")
        loss = MultiClassTgLoss(dataset_config["num_labels"])
        # loss = CoSENTLoss(dataset_config["num_labels"])
        # loss = MultiClassFocalLoss()

        # 设置优化器
        trainer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        # scheduler = transformers.get_linear_schedule_with_warmup(trainer, num_warmup_steps=warmup_steps,
        # num_training_steps=int(steps_per_epoch * num_epochs))

        net.to(devices[0])

        # 训练模型
        test_acc = train(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
        test_acc_list.append(test_acc)
    print(test_acc_list)
    print(sum(test_acc_list) / len(test_acc_list))
