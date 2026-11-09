import math
import torch

''' This is mirror of the model.py file, representing the exact same model, but with some optimizations
for traning and inference:
    - scaled_dot_product_attention
    - got rid of head class, and instead using a single linear layer to compute q,k,v
    '''

class MultiHead(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_k=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_v = d_model // n_heads
        self.d_k = d_k if d_k is not None else self.d_v
        self.W_qkv = torch.nn.Linear(d_model, n_heads * (2 * self.d_k + self.d_v))
        self.W_o = torch.nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, X):
        B, T, _ = X.shape
        hk, hv = self.h * self.d_k, self.h * self.d_v
        q, k, v = self.W_qkv(X).split([hk, hk, hv], dim=-1)
        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)   # (B,h,T,d_k)
        k = k.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_v).transpose(1, 2)   # (B,h,T,d_v)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.W_o(out.transpose(1, 2).reshape(B, T, hv))


class MLP(torch.nn.Module):

    def __init__(self, d_model):
            
        super().__init__()
        self.W_ff1 = torch.nn.Linear(d_model, 4*d_model)
        self.W_ff2 = torch.nn.Linear(4*d_model, d_model)

    def forward(self, X):

        linear_1 = self.W_ff1(X)
        non_linearity = torch.nn.functional.gelu(linear_1)
        linear_2 = self.W_ff2(non_linearity)
        return linear_2


class Block(torch.nn.Module):
     
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()
        self.att = MultiHead(d_model, n_heads, d_k)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = torch.nn.LayerNorm(d_model)

    def forward(self, X):

        X = X + self.att(self.ln_1(X))
        X = X + self.mlp(self.ln_2(X))
        return X


class Decoder_Only_Transformer(torch.nn.Module):

    def __init__(self, vocab_size, block_size, d_model, n_heads = 1, d_k = None, n_layers = 1):

        super().__init__()
        self.block_size = block_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model) 
        self.blocks = torch.nn.ModuleList([Block(d_model, n_heads, d_k) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_encoding", self.sinusoidal_pe(block_size, d_model))
        self.loss = torch.nn.CrossEntropyLoss()

    
    @staticmethod
    def sinusoidal_pe(block_size, d_model):
        pe = torch.zeros(block_size, d_model)
        for pos in range(block_size):
            for i in range(0, d_model, 2):
                denom = 10000 ** (i / d_model)
                pe[pos, i]     = math.sin(pos / denom)
                pe[pos, i + 1] = math.cos(pos / denom)
        return pe

    
    
    def forward(self, X, Y = None):

        B, T = X.shape
        X_e = self.embedding(X) + self.pos_encoding[:T] #(B,T,d_model)
        for block in self.blocks:
            X_e = block(X_e)
        X_e = self.ln_f(X_e) #(B,T,d_model)
        logits = self.linear(X_e) #(B,T,vocab_size)
        loss = None
        if Y is not None:
            loss = self.loss(logits.reshape(B*T,-1), Y.reshape(B*T,))
        
        return logits, loss


    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        block_size = self.pos_encoding.shape[0]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            yield next_tok.item()
