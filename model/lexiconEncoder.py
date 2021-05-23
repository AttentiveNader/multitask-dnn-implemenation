import torch
from torch import nn

"""
this implementation of the lexicon encoder is based on the one from BERT https://arxiv.org/abs/1810.04805
the positionalEncoder formula can be found in the Attention is all you need paper https://arxiv.org/abs/1706.03762
PE(pos,2i)   =  sin(pos/10000.exp(2i/d_model))
PE(pos,2i+1) =  cos(pos/10000.exp(2i/d_model))
"""
"""
todo 
figure out how the positional encoding works ??!! XD
use vmap on the div_term
check the shape broadcasting 

"""


class PositionalEncoder(nn.Module):

    def __init__(self,d_model:int,max_len:int=512):
        super().__init__()
        pe = torch.zeros((max_len,d_model)).float()
        div_term = torch.arange(0,d_model).unsqueeze(0).float()
        pos = torch.arange(0,max_len).unsqueeze(1).float()

        self.register_buffer()



class LexiconEncoder(nn.Module):

    def __init__(self, maxSeqLength: int, vocab_size: int=20):
        super().__init__()






