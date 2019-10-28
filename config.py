# coding:utf-8
# Bert Config

__name__ = ['bert_config']

class BertConfig(object):
    def __init__(self):
        # train
        self.learning_rate = 1e-3

        # model
        self.vocab_size = 32
        self.embedding_size = 32
        self.hidden_size = 16
        self.max_positional_embeddings = 30

        # initializer
        self.initializer_range = 0.01

        # global
        self.data_path = 'data/train.data'

bert_config = BertConfig()