import csv
import os
import json
import re
import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd
import tqdm

from .utils import word_repetition

_HEADER = ["text", "category"]

PATTERNS = {
    "train": "../../data/datasets/hwu/KFold_1/trainset/{f}",
    "test": "../../data/datasets/hwu/KFold_1/testset/csv/{f}"
}

LIST_OF_FILES = (
    'alarm_query.csv\nalarm_remove.csv\nalarm_set.csv\naudio_volum'
    'e_down.csv\naudio_volume_mute.csv\naudio_volume_up.csv\ncalend'
    'ar_query.csv\t\ncalendar_remove.csv\t\ncalendar_set.csv\t\ncoo'
    'king_recipe.csv\t\ndatetime_convert.csv\t\ndatetime_query.csv'
    '\t\nemail_addcontact.csv\t\nemail_query.csv\t\nemail_querycon'
    'tact.csv\t\nemail_sendemail.csv\t\ngeneral_affirm.csv\t\ngener'
    'al_commandstop.csv\t\ngeneral_confirm.csv\t\ngeneral_dontcare.'
    'csv\t\ngeneral_explain.csv\t\ngeneral_joke.csv\t\ngeneral_neg'
    'ate.csv\t\ngeneral_praise.csv\t\ngeneral_quirky.csv\t\ngenera'
    'l_repeat.csv\t\niot_cleaning.csv\t\niot_coffee.csv\t\niot_hue'
    '_lightchange.csv\t\niot_hue_lightdim.csv\t\niot_hue_lightoff.'
    'csv\t\niot_hue_lighton.csv\t\niot_hue_lightup.csv\t\niot_wemo_'
    'off.csv\t\niot_wemo_on.csv\t\nlists_createoradd.csv\t\nlists_'
    'query.csv\t\nlists_remove.csv\t\nmusic_likeness.csv\t\nmusic_q'
    'uery.csv\t\nmusic_settings.csv\t\nnews_query.csv\t\nplay_audio'
    'book.csv\t\nplay_game.csv\t\nplay_music.csv\t\nplay_podcasts.'
    'csv\t\nplay_radio.csv\t\nqa_currency.csv\t\nqa_definition.csv'
    '\t\nqa_factoid.csv\t\nqa_maths.csv\t\nqa_stock.csv\t\nrecomme'
    'ndation_events.csv\t\nrecommendation_locations.csv\t\nrecomme'
    'ndation_movies.csv\t\nsocial_post.csv\t\nsocial_query.csv\t\n'
    'takeaway_order.csv\t\ntakeaway_query.csv\t\ntransport_query.c'
    'sv\t\ntransport_taxi.csv\t\ntransport_ticket.csv\t\ntransport'
    '_traffic.csv\t\nweather_query.csv\t'.split())


def read_hwu(dataset_config, is_train, shot=None):
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
    # with open('../data/preprocessed/hwu/labels_dict.pkl', 'wb') as f:
    #     pickle.dump(dict_data, f)

    df = pd.read_csv(file_name)
    labels = list(df.loc[:, 'category'])

    for i, label in enumerate(labels):
        labels[i] = dict_data[label]
    sentences = list(df.loc[:, 'text'])

    return sentences, labels


class HwuDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, encoder_config, num_steps, need_repetition=False, is_roberta=False):
        self.max_length = num_steps
        self.is_roberta = is_roberta
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_config["model"])
        self.sentences = dataset[0]
        self.labels = torch.tensor(dataset[1])
        self.need_repetition = need_repetition
        print('read ' + str(len(self.sentences)) + ' examples')

    def __getitem__(self, idx):
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


def load_data_hwu(batch_size, encoder_config, dataset_config, num_steps=75, need_repetition=False, shot=None, is_roberta=False):
    train_data = read_hwu(dataset_config, True, shot)
    test_data = read_hwu(dataset_config, False)
    train_set = HwuDataset(train_data, encoder_config, num_steps, need_repetition, is_roberta)
    test_set = HwuDataset(test_data, encoder_config, num_steps, False, is_roberta)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                             collate_fn=train_set.smart_batching_collate)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False,
                                            collate_fn=test_set.smart_batching_collate)
    return train_iter, test_iter


def _get_final_rows(set_name: str):
    final_rows = [_HEADER]
    for f in tqdm(LIST_OF_FILES):
        final_rows += _get_category_rows(f, set_name)
    return final_rows


def _get_category_rows(fname: str, set_name: str):
    pattern = PATTERNS[set_name]
    url = pattern.format(f=fname)

    df = pd.read_csv(url, sep=';')

    rows = []
    for index, row in df.iterrows():
        text = row["answer_from_anno"]
        category = f"{row['scenario']}_{row['intent']}"
        rows.append([text, category])
    return rows


def _write_data_into_file(path, rows):
    with open(path, "w") as data_file:
        writer = csv.writer(data_file, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def creat_data():
    print("Getting train data")
    train_rows = _get_final_rows(set_name="train")
    _write_data_into_file(
        path=os.path.join("../../data/datasets/hwu/", "train.csv"),
        rows=train_rows
    )

    print("Getting test data")
    test_rows = _get_final_rows(set_name="test")
    _write_data_into_file(
        path=os.path.join("../../data/datasets/hwu/", "test.csv"),
        rows=test_rows
    )

    print("Creating categories.json file")
    _, train_cats = zip(*train_rows[1:])
    _, test_cats = zip(*test_rows[1:])
    categories = sorted(list(
        set(train_cats) | set(test_cats)
    ))
    with open(os.path.join("../../data/datasets/hwu/", "categories.json"), "w") as f:
        json.dump(categories, f)


# def creat_csv():
#     df = pd.read_csv("../data/dataset/hwu/NLU-Data-Home-Domain-Annotated-All.csv", sep=';')
#     df = df.sample(frac=1).reset_index(drop=True)
#
#     train_len = len(df) * 0.8
#     train = df.loc[:train_len, ['answer', 'scenario']]
#     test = df.loc[train_len:, ['answer', 'scenario']]
#
#     # train.to_csv("../data/dataset/hwu/train.csv", sep='\t', index=False)
#     # test.to_csv("../data/dataset/hwu/test.csv", sep='\t', index=False)
#
#     labels = list(df.loc[:, 'scenario'])
#     categories = sorted(list(
#         set(labels)
#     ))
#     with open("../data/dataset/hwu/categories.json", 'w', encoding='utf-8') as f:
#         json.dump(categories, f)


if '__main__' == __name__:
    dataset_config_path = "./../../config/mlp/hwu.json"
    with open(os.path.normpath(dataset_config_path), 'r') as config_file:
        dataset_config = json.load(config_file)

    train_data = read_hwu(dataset_config, is_train=True)
    print('train data size =', len(train_data[0]))
    for x0, y in zip(train_data[0][:3], train_data[1][:3]):
        print('句子：', x0)
        print('标签：', y)

    test_data = read_hwu(dataset_config, is_train=False)
    print('test data size =', len(test_data[0]))
    for x0, y in zip(train_data[0][:3], train_data[1][:3]):
        print('句子：', x0)
        print('标签：', y)
