import torch 
import math 
import torch.nn as nn
import torch.optim as optim 

x = torch.randn(3, 4)
def asker (x, W_Q, W_K, W_V): 

    Q = W_Q(x)
    K = W_K(x) 
    V = W_V(x) 

    scores = Q @ K.T 
    scores = scores / math.sqrt(K.shape[-1])

    attetion = torch.softmax(scores, dim =-1)
    M = attetion @ V 


class GPT_Astra(nn.Module):
    def __init__(self, d_model=4, num_heads=2):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 

        self.W_Q1 = nn.Linear(d_model, self.head_dim)
        self.W_K1 = nn.Linear(d_model, self.head_dim)
        self.W_V1 = nn.Linear(d_model, self.head_dim)

        self.W_Q2 = nn.Linear(d_model, self.head_dim)
        self.W_K2 = nn.Linear(d_model, self.head_dim)
        self.W_V2 = nn.Linear(d_model, self.head_dim)
        
        self.W_O = nn.Linear(self.num_heads * self.head_dim, d_model)

        self.norm = nn.LayerNorm(d_model)

    def asker(self, x, W_Q, W_K, W_V):

        return output 

    def forward(self, x):
        head1 = asker(x, self.W_Q1, self.W_K1, self.W_V1)
        head2 = asker(x, self.W_Q2, self.W_K2, self.W_V2)

        concat = torch.cat([head1, head2], dim=-1)
        output = self.W_O(concat)

        residual = x 
        output = residual + output 
        output = self.norm(output)
        return output


def askerFFN (x, W1, W2): 

    AS = W1(x) 
    ASB = F.gelu(AS)
    ASV = W2(x)

    return ASV

class FFN(nn.Module):
    def __init__(self, d_model=4, hidden_dim =16): 
        super().__init__()

        self.W1 = nn.Linear(d_model, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        output = askerFFN(x, self.W1, self.W2)

        residual = x 
        output = residual + output 
        output = self.norm(output)
        return output





model = GPT_Astra()
ffn_model = FFN()

print("--- ATTENTION WEIGHTS ---")
for name, param in model.named_parameters():
    print(name, param.shape)

print("--- FFN WEIGHTS ---")
for name, param in ffn_model.named_parameters():
    print(name, param.shape)

x = torch.randn(3, 4)

attention_output = model(x)
print("Attention:", attention_output.shape)

final_output = ffn_model(attention_output)
print("FFN:", final_output.shape)

print("Valid:", torch.isfinite(final_output).all())
