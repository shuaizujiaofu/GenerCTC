from torch import nn
from transformers import AutoModel
from .utils import mean_pooling, max_pooling


class Mlp(nn.Module):
    def __init__(self, encoder_config, num_labels, pool='cls_before_pooler'):
        super(Mlp, self).__init__()
        self.hidden_size = encoder_config["embedding_size"]
        self.pool = pool
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.encoding = AutoModel.from_pretrained(encoder_config["model"])
        self.is_roberta = self.encoding.config.model_type == "roberta"

    def forward(self, X):
        # encode
        bert_output = self.encode(X)

        # pool
        pooled_output = self.pooled(bert_output, X)

        # dropout
        # pooled_output = self.dropout(pooled_output)

        # classify
        logits = self.classifier(pooled_output)

        return logits

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
