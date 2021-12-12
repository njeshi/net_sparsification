import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        '''
        Parameters
        ----------
        input_ids : 
            Tensor of shape [B, T] containing token ids of sequences
        attention_mask : 
            Tensor of shape [B, T] containing attention masks to be used to 
            avoid contibution of PAD tokens (where B is the batch 
            size and T is the input length)
        '''
        # Feed the input to Bert model to obtain contextualized representations
        reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        cls_reps = reps[:, 0]
        # cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits
