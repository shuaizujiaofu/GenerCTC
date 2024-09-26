import csv
import os
import json
import re
import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd

from .utils import word_repetition


def read_clinc(dataset_config, is_train, shot=None):
    data_dir = dataset_config["data_path"]
    if is_train:
        if shot is None:
            file_name = data_dir + '/train.csv'
        elif shot == 5:
            file_name = data_dir + '/train_5.csv'
        elif shot == 10:
            file_name = data_dir + '/train_10.csv'
        else:
            raise 'This shot does not exist'
    else:
        file_name = data_dir + '/test.csv'

    # 读取json文件内容,返回字典格式
    with open(dataset_config["categories_path"], 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    dict_data = {}
    for i, arg in enumerate(json_data):
        # arg = arg.replace("_", " ")
        dict_data[arg] = i

    # 保存dict
    # with open('../data/preprocessed/clinc/labels_dict.pkl', 'wb') as f:
    #     pickle.dump(dict_data, f)

    df = pd.read_csv(file_name, encoding='gbk')
    labels = list(df.loc[:, 'category'])

    for i, label in enumerate(labels):
        labels[i] = dict_data[label]
    sentences = list(df.loc[:, 'text'])

    return sentences, labels


class ClincDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, encoder_config, num_steps, need_repetition=False, is_roberta=False):
        self.max_length = num_steps
        self.is_roberta = is_roberta
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_config["model"])
        self.sentences = dataset[0]
        self.labels = torch.tensor(dataset[1])
        self.need_repetition = need_repetition
        print('read ' + str(len(self.sentences)) + ' examples')

    def __getitem__(self, idx):
        # sentence = self.sentences[idx]
        # all_sentence_tokens = self.tokenizer(sentence, padding='max_length', truncation=True,
        #                                     max_length=self.max_length, return_tensors='pt')
        # sentence_ids = all_sentence_tokens.input_ids.squeeze(0)
        # sentence_token_type_ids = all_sentence_tokens.token_type_ids.squeeze(0)
        # sentence_attention_mask = all_sentence_tokens.attention_mask.squeeze(0)
        # return (sentence_ids, sentence_token_type_ids, sentence_attention_mask), self.labels[idx]

        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentences)

    def tokenize(self, texts):
        sentence_features = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt',
                                           max_length=self.max_length)
        sentence_ids = sentence_features.input_ids.squeeze(0)
        sentence_attention_mask = sentence_features.attention_mask.squeeze(0)

        if not self.is_roberta:
            sentence_token_type_ids = sentence_features.token_type_ids.squeeze(0)
            return [sentence_ids, sentence_token_type_ids, sentence_attention_mask]
        else:
            return [sentence_ids, sentence_attention_mask]

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = []
        labels = []

        for example in batch:
            texts.append(example[0])
            labels.append(example[1])

        labels = torch.tensor(labels)

        sentence_features = self.tokenize(texts)

        if self.need_repetition:
            dst_text = word_repetition(texts, dup_rate=0.20)
            dst_sentence_features = self.tokenize(dst_text)

            return [sentence_features, dst_sentence_features], labels
        else:
            return sentence_features, labels


def load_data_clinc(batch_size, encoder_config, dataset_config, num_steps=75, need_repetition=False, shot=None,
                    is_roberta=False):
    train_data = read_clinc(dataset_config, True, shot)
    test_data = read_clinc(dataset_config, False)
    train_set = ClincDataset(train_data, encoder_config, num_steps, need_repetition, is_roberta)
    test_set = ClincDataset(test_data, encoder_config, num_steps, False, is_roberta)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                             collate_fn=train_set.smart_batching_collate)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True,
                                            collate_fn=test_set.smart_batching_collate)
    return train_iter, test_iter


def creat_categories():
    with open("../data/dataset/clinc/dataset.json", 'r', encoding='utf-8') as fp:
        targets = []
        data = [json.loads(line) for line in fp][0]['CLINC150']
        for domain, sentence in data.items():
            for group in sentence:
                targets.append(group[1][0])

    label_dict = set(targets)
    print(label_dict)

    with open("../data/dataset/clinc/categories.json", 'w', encoding='utf-8') as f:
        f.writelines(str(label_dict))


if '__main__' == __name__:
    dataset_config_path = "./../../config/mlp/clinc.json"
    with open(os.path.normpath(dataset_config_path), 'r') as config_file:
        dataset_config = json.load(config_file)

    train_data = read_clinc(dataset_config, is_train=True)
    print('train data size =', len(train_data[0]))
    for x0, y in zip(train_data[0][:3], train_data[1][:3]):
        print('句子：', x0)
        print('标签：', y)

    test_data = read_clinc(dataset_config, is_train=False)
    print('test data size =', len(test_data[0]))
    for x0, y in zip(train_data[0][:3], train_data[1][:3]):
        print('句子：', x0)
        print('标签：', y)
