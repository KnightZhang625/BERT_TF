# coding:utf-8
# Bert Config

__name__ = ['bert_config']

class BertConfig(object):
    def __init__(self):
        # train
        self.learning_rate = 1e-3

        # model
        self.vocab_size = 7819
        self.embedding_size = 32
        self.hidden_size = 64
        self.max_positional_embeddings = 30
        self.token_type_vocab_size = 0
        self.pre_positional_embedding_type = 'normal'
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1 
        self.num_hidden_layers = int(2)
        self.num_attention_heads = int(2)
        self.intermediate_size = 32

        # initializer
        self.initializer_range = 0.01

        # global
        self.data_path = 'data/train.data'
        self.model_dir = 'models/'
        self.init_checkpoint = None
        self.batch_size = 2
        self.num_train_steps = 100

bert_config = BertConfig()