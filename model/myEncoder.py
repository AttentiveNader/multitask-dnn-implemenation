from torch import nn
import torch


"""
this implementation depends highly on torch.einsum 
"""



# todo
"""
figure out  whether the projection weight matrices for q,v and k are different for every head 
or it's just a big weight matrix before splitting into heads(am using the latter for now)

also implement the feedforward network
"""



class MultiHeadAttention(nn.Module):
    def __int__(self,d_model:int,n_heads:int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.queries = nn.Linear(self.d_model,self.d_model,bias=False)
        self.keys = nn.Linear(self.d_model,self.d_model,bias=False)
        self.values = nn.Linear(self.d_model,self.d_model,bias=False)
        self.fc_out = nn.Linear(d_model,d_model)

    def forward(self,queries,keys,values,mask)->torch.Tensor:

        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        queries = queries.reshape(queries.shape[0],queries.shape[1],self.n_heads,self.d_head)
        keys = keys.reshape(keys.shape[0],keys.shape[1],self.n_heads,self.d_head)
        values = values.reshape(values.shape[0],values.shape[1],self.n_heads,self.d_head)

        attn = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])

        if mask is not None:
            attn = attn.masked_fill(mask == 0,float('1e-18'))
        attn = torch.softmax(attn/(self.d_model**0.5),dim=3)

        # out = torch.einsum('nhqk,nvhd->nvhd')
        out = torch.einsum('nhql,nlhd->nqhd'[attn,values]).reshape(values.shape[0],values.shape[1],self.d_model)

        out = self.fc_out(out)

        return out


class EncoderLayer(nn.Module):

    def __int__(self,d_model:int,n_heads):
        super(EncoderLayer, self).__int__()
        self.multiHeadAttention = MultiHeadAttention(d_model,n_heads)