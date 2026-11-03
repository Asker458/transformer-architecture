import torch
import torch.nn as nn
import torch.optim as optim 

class Asker_LLM (nn.Module): 
    def __init__ (self, размер_эмбедина, количство_токенов):
        super().__init__()

        self.эмбединг = nn.Embedding(num_embeddings = количство_токенов, embedding_dim= 8)
        self.позиция_эмбединга = nn.Embedding(макс_длина, размер_эмбедина)

        self.внимание = nn.MultiheadAttention(embed_dim = размер_эмбедина, num_heads = количство_токенов, batch_first=True) 

        self.оптимизация = nn.LayerNorm(normalized = размер_эмбедина)  
     