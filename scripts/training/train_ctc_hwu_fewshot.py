import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange
import os
import json
from scipy.stats import pearsonr, spearmanr

from model.model_ctc import DecomposableClassification
from scripts.preprocessing.preprocess_hwu import load_data_hwu
from model.loss_cosent import CoSENTLoss
import utils


def train_batch(net, X, y, loss, trainer, mode, num_labels, devices, scheduler=None):
    def move_to(obj, device):
        if isinstance(obj, list):
            res = []
            for v in obj:
                res.append(move_to(v, device))
            return res
        else:
            return obj.to(device)

    X = move_to(X, devices[0])
    y = y.to(devices[0])

    net.train()

    if mode == 'pre-train':
        # labels_mask = util.onehot_labeling(y, num_labels)
        labels_mask = torch.eye(len(y)).to(devices[0])

        x_sim, x_adv_sim = net(X, mode)
        l = (loss(x_sim, labels_mask) + loss(x_adv_sim, labels_mask)) / 2
        l.sum().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        trainer.step()
        trainer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        train_loss_sum = l.sum()
        train_pearson_sum = 0
        if labels_mask.shape[0] > 1:
            for i in range(labels_mask.shape[0]):
                train_pearson, _ = pearsonr(labels_mask[i].tolist(), x_sim[i].cpu().detach().numpy())
                train_pearson_adv, _ = pearsonr(labels_mask[i].tolist(), x_adv_sim[i].cpu().detach().numpy())
                train_pearson_sum += (train_pearson + train_pearson_adv) / 2
        return train_loss_sum, train_pearson_sum

    elif mode == 'init':
        pred = net(X, mode)
        l = loss(pred, y)
        l.sum().backward()
        trainer.step()
        trainer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        train_loss_sum = l.sum()
        train_acc_sum = d2l.accuracy(pred, y)
        return train_loss_sum, train_acc_sum

    elif mode == 'fine-tune':
        pred, pred_adv = net(X, mode)
        l = (loss(pred, y) + loss(pred_adv, y)) / 2
        l.sum().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        trainer.step()
        trainer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        train_loss_sum = l.sum()
        train_acc_sum = d2l.accuracy(pred, y)
        return train_loss_sum, train_acc_sum

    else:
        raise 'This mode does not exist'


def train(net, train_iter, test_iter, loss, trainer, num_epochs, dataset_config, mode, devices=d2l.try_all_gpus()):
    num_batches = len(train_iter)
    best_acc = 0

    if mode == "pre-train":
        save_path = dataset_config["save_path"] + "pre-train" + ".pth.tar"
    elif mode == "init":
        save_path = dataset_config["save_path"] + "init" + ".pth.tar"
    else:
        save_path = dataset_config["save_path"] + "fine-tune" + ".pth.tar"
    print("save_path = ", save_path)

    writer = SummaryWriter(dataset_config["tensorboard_path"])
    for epoch in trange(num_epochs, desc="Epoch", disable=False):
        # Sum of training loss, sum of training spearman, no. of examples, no. of predictions
        metric = d2l.Accumulator(4)
        train_iterators = iter(train_iter)
        for i in trange(len(train_iter), desc="Iteration", smoothing=0.05, disable=True):
            data = next(train_iterators)
            features, labels = data
            l, acc = train_batch(net, features, labels, loss, trainer, mode, dataset_config["num_labels"], devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if i == num_batches - 1:
                print("train loss =", metric[0] / metric[2])
                writer.add_scalar('train/loss', metric[0] / metric[2], int(epoch + (i + 1) / num_batches))
                if mode == "pre-train":
                    print("train pearson =", metric[1] / metric[2])
                    writer.add_scalar('train/pearson', metric[1] / metric[2], int(epoch + (i + 1) / num_batches))
                else:
                    print("train acc =", metric[1] / metric[3])
                    writer.add_scalar('train/acc', metric[1] / metric[3], int(epoch + (i + 1) / num_batches))
                metric.reset()
        if mode == 'pre-train':
            test_acc = utils.evaluate_pearson_gpu(net, test_iter, dataset_config["num_labels"])
        else:
            test_acc = utils.evaluate_accuracy_gpu(net, test_iter, mode="test_acc")
        writer.add_scalar('test/acc', test_acc, epoch + 1)
        print("test acc =", test_acc)
        # 保存模型
        if test_acc - best_acc > 0.0001:
            best_acc = test_acc
            torch.save({"epoch": epoch,
                        "model": net.state_dict(),
                        "best_acc": best_acc},
                       save_path)
            print("save model success")
    print(f'best acc {best_acc:.4f}')

    return best_acc


if __name__ == '__main__':
    encoder_config_path = "./../../config/bert_mini.json"
    with open(os.path.normpath(encoder_config_path), 'r') as config_file:
        encoder_config = json.load(config_file)
    dataset_config_path = "./../../config/ctc/hwu.json"
    with open(os.path.normpath(dataset_config_path), 'r') as config_file:
        dataset_config = json.load(config_file)
    devices = d2l.try_all_gpus()

    # 设置超参数
    batch_size, num_steps = 64, 75
    pool = "cls_before_pooler"
    fewshot_num = 10

    # 定义训练阶段
    test_acc_list = []
    for i in range(4):
        for mode in ["pre-train", "init", "fine-tune"]:
            if mode == "pre-train":
                # 创建模型
                net = DecomposableClassification(encoder_config["model"], dataset_config["num_labels"], pool=pool)
                # 读取数据集
                train_iter, test_iter = load_data_hwu(batch_size, encoder_config, dataset_config, num_steps, True,
                                                      shot=fewshot_num,
                                                      is_roberta=net.is_roberta)

                lr, num_epochs = 1e-5, 1

            elif mode == 'init':
                # 创建模型
                net = DecomposableClassification(encoder_config["model"], dataset_config["num_labels"], pool=pool)

                # 读取数据集
                train_iter, test_iter = load_data_hwu(batch_size, encoder_config, dataset_config, num_steps, False,
                                                      fewshot_num,
                                                      net.is_roberta)

                # 加载预训练模型
                pretrained_file = dataset_config["save_path"] + 'pre-train.pth.tar'
                checkpoint = torch.load(pretrained_file, map_location=torch.device('cpu'))
                net.load_state_dict(checkpoint["model"], strict=True)

                for name, param in net.named_parameters():
                    if "encoding" in name:
                        param.requires_grad = False

                # creat label pc embedding
                label_pc = utils.get_pc(net, train_iter)

                # save label pc
                # if encoder_config["embedding_size"] == 768:
                #     pc_path = '../data/preprocessed/hwu/labels_pc_bert.pkl'
                # else:
                #     pc_path = '../data/preprocessed/hwu/labels_pc_minilm.pkl'
                # with open(pc_path, 'wb') as f:
                #     pickle.dump(label_pc, f)
                # label_pc = util.load_pc(pc_path)

                label_embeddings = nn.Parameter(label_pc.unsqueeze(0), requires_grad=True)
                net.label_embeddings.data.copy_(label_embeddings)
                net.label_embeddings.requires_grad = True

                lr, num_epochs = 9e-4, 30

            elif mode == 'fine-tune':
                # 创建模型
                net = DecomposableClassification(encoder_config["model"], dataset_config["num_labels"], pool=pool)
                # 读取数据集
                train_iter, test_iter = load_data_hwu(batch_size, encoder_config, dataset_config, num_steps, True,
                                                      fewshot_num, net.is_roberta)

                # 加载预训练模型
                pretrained_file = dataset_config["save_path"] + 'init.pth.tar'
                checkpoint = torch.load(pretrained_file, map_location=torch.device('cpu'))
                net.load_state_dict(checkpoint["model"], strict=True)

                lr, num_epochs = 3e-5, 150

            else:
                raise 'This mode does not exist'

            # 定义参数
            param_optimizer = list(net.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            # warmup_steps = math.ceil(len(train_iter) * num_epochs * 0.1)  # 10% of train data for warm-up
            # steps_per_epoch = min([len(train_iter) for train_iter in train_iter])

            # 定义模块
            loss = CoSENTLoss(dataset_config["num_labels"])
            # loss = nn.CrossEntropyLoss(reduction="none")
            trainer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
            net.to(devices[0])
            # scheduler = transformers.get_linear_schedule_with_warmup(trainer, num_warmup_steps=warmup_steps,
            # num_training_steps=int(steps_per_epoch * num_epochs))

            # 训练模型
            if mode == "fine-tune":
                test_acc = train(net, train_iter, test_iter, loss, trainer, num_epochs, dataset_config, mode, devices)
                test_acc_list.append(test_acc)
            else:
                _ = train(net, train_iter, test_iter, loss, trainer, num_epochs, dataset_config, mode, devices)
    print("test_acc_list = ", test_acc_list)
    print("5 mean acc = ", sum(test_acc_list) / len(test_acc_list))
