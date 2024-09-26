import torch
from torch import nn
from transformers import AutoModel
from .utils import mean_pooling, max_pooling


class DecomposableClassification(nn.Module):
    def __init__(self, encoding_path, num_labels, pool="cls_before_pooler"):
        super(DecomposableClassification, self).__init__()
        self.encoding = AutoModel.from_pretrained(encoding_path)
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, self.encoding.config.hidden_size).unsqueeze(0), requires_grad=True)
        self.pretrain_mlp = nn.Sequential(
            nn.Linear(self.encoding.config.hidden_size, self.encoding.config.hidden_size),
            nn.Tanh()
        )
        self.pool = pool
        self.dropout = nn.Dropout(0.1)
        self.is_roberta = self.encoding.config.model_type == "roberta"

    def forward(self, X, mode):
        if mode == 'pre-train':
            sentence_features, dst_sentence_features = X

            # encode
            bert_output = self.encode(sentence_features)
            bert_adv_output = self.encode(dst_sentence_features)

            # pool
            pooled_output = self.pooled(bert_output, sentence_features)
            pooled_adv_output = self.pooled(bert_adv_output, dst_sentence_features)

            # 经过mlp转换
            pooled_output = self.pretrain_mlp(pooled_output)
            pooled_adv_output = self.pretrain_mlp(pooled_adv_output)

            # l2正则化
            pooled_output_norm = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
            pooled_adv_output_norm = torch.nn.functional.normalize(pooled_adv_output, p=2, dim=1)

            # 计算句子间的相似度
            x_sim = torch.mm(pooled_output_norm, pooled_adv_output_norm.transpose(0, 1))
            x_adv_sim = torch.mm(pooled_adv_output_norm, pooled_output_norm.transpose(0, 1))

            return x_sim, x_adv_sim

        elif mode == 'init':
            with torch.no_grad():
                # encode
                bert_output = self.encode(X)

                # pool
                pooled_output = self.pooled(bert_output, X)

                # pooled_output = self.dropout(pooled_output)

                # 经过mlp转换
                # pooled_output = self.pretrain_mlp(pooled_output)

                pooled_output = pooled_output.unsqueeze(1)
                pooled_output_norm = torch.nn.functional.normalize(pooled_output, p=2, dim=2)

            label_embeddings = self.label_embeddings.repeat(pooled_output.shape[0], 1, 1)
            label_embeddings_norm = torch.nn.functional.normalize(label_embeddings, p=2, dim=2)

            # 所有分类的相似度
            logits = torch.bmm(pooled_output_norm, label_embeddings_norm.permute(0, 2, 1))

            return logits.squeeze(dim=1)

        elif mode == 'fine-tune':
            sentence_features, dst_sentence_features = X

            # encode
            bert_output = self.encode(sentence_features)
            bert_adv_output = self.encode(dst_sentence_features)

            # pool
            pooled_output = self.pooled(bert_output, sentence_features)
            pooled_adv_output = self.pooled(bert_adv_output, dst_sentence_features)

            # 经过mlp转换
            # pooled_output = self.pretrain_mlp(pooled_output)

            # pooled_output = self.dropout(pooled_output)
            # pooled_adv_output = self.dropout(pooled_adv_output)

            pooled_output = pooled_output.unsqueeze(1)
            pooled_output_norm = torch.nn.functional.normalize(pooled_output, p=2, dim=2)
            pooled_adv_output = pooled_adv_output.unsqueeze(1)
            pooled_adv_output_norm = torch.nn.functional.normalize(pooled_adv_output, p=2, dim=2)

            label_embeddings = self.label_embeddings.repeat(pooled_output.shape[0], 1, 1)
            label_embeddings_norm = torch.nn.functional.normalize(label_embeddings, p=2, dim=2)

            # 所有分类的相似度
            logits = torch.bmm(pooled_output_norm, label_embeddings_norm.permute(0, 2, 1))
            logits_adv = torch.bmm(pooled_adv_output_norm, label_embeddings_norm.permute(0, 2, 1))

            return logits.squeeze(dim=1), logits_adv.squeeze(dim=1)

        else:
            raise 'This mode does not exist'

    def test_acc(self, X):
        with torch.no_grad():
            # encode
            bert_output = self.encode(X)

            # pool
            pooled_output = self.pooled(bert_output, X)

            pooled_output = pooled_output.unsqueeze(1)
            pooled_output_norm = torch.nn.functional.normalize(pooled_output, p=2, dim=2)

            label_embeddings = self.label_embeddings.repeat(pooled_output.shape[0], 1, 1)
            label_embeddings_norm = torch.nn.functional.normalize(label_embeddings, p=2, dim=2)

            # 所有分类的相似度
            logits = torch.bmm(pooled_output_norm, label_embeddings_norm.permute(0, 2, 1))

            return logits.squeeze(dim=1)

    def test_pearson(self, X):
        with torch.no_grad():
            # encode
            bert_output = self.encode(X)

            # pool
            pooled_output = self.pooled(bert_output, X)

            # l2正则化
            pooled_output_norm = torch.nn.functional.normalize(pooled_output, p=2, dim=1)

            # 计算句子间的相似度
            x_sim = torch.mm(pooled_output_norm, pooled_output_norm.transpose(0, 1))

            return x_sim

    def encode(self, X):
        if self.is_roberta:
            return self.roberta_encode(X)
        else:
            return self.bert_encode(X)

    def bert_encode(self, X):
        sentence_ids, sentence_token_type_ids, sentence_attention_mask = X

        # batch_size == 1
        if len(sentence_ids.shape) == 1:
            sentence_ids = sentence_ids.unsqueeze(0)
            sentence_token_type_ids = sentence_token_type_ids.unsqueeze(0)
            sentence_attention_mask = sentence_attention_mask.unsqueeze(0)
        bert_output = self.encoding(sentence_ids, sentence_token_type_ids, sentence_attention_mask)

        return bert_output

    def roberta_encode(self, X):
        sentence_ids, sentence_attention_mask = X

        # batch_size == 1
        if len(sentence_ids.shape) == 1:
            sentence_ids = sentence_ids.unsqueeze(0)
            sentence_attention_mask = sentence_attention_mask.unsqueeze(0)
        bert_output = self.encoding(sentence_ids, sentence_attention_mask)

        return bert_output

    def pooled(self, bert_output, X):
        if self.is_roberta:
            _, sentence_attention_mask = X
        else:
            _, _, sentence_attention_mask = X

        if self.pool == "pooler":
            pooled_output = bert_output.pooler_output
        elif self.pool == "cls_before_pooler":
            pooled_output = bert_output.last_hidden_state[:, 0]
        elif self.pool == "last_avg":
            bert_output = bert_output[0]
            pooled_output = mean_pooling(bert_output, sentence_attention_mask)
        elif self.pool == "last_max":
            bert_output = bert_output[0]
            pooled_output = max_pooling(bert_output, sentence_attention_mask)
        elif self.pool == "first_last_avg":
            bert_output = bert_output.hidden_states
            pooled_output = (mean_pooling(bert_output[1], sentence_attention_mask)
                             + mean_pooling(bert_output[-1], sentence_attention_mask)) \
                            / 2
        else:
            raise 'This pool method does not exist'

        # pooled_output = self.dropout(pooled_output)

        return pooled_output

