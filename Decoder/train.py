import torch

from model import Decoder_Only_Transformer

PERMITED = set("abcdefghijklmnñopqrstuvwxyzáéíóúü"
               "ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ"
               " \n.,;:¡!¿?-'\"()")


class DataLoader:
    
    def __init__(self, n_batches, batch_size, block_size):
        if block_size < 1:
            raise ValueError(f"block_size {block_size} must be at least 1")
        if batch_size < 1:
            raise ValueError(f"batch_size {batch_size} must be at least 1")
        if n_batches < 1:
            raise ValueError(f"n_batches {n_batches} must be at least 1")
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_data, self.val_data, self.chars = self.load_data("Obras_completas_ilustradas_Federico_Garcia_Lorca_djvu.txt")

    def __iter__(self):
        return self.batches(self.train_data, n_batches=self.n_batches, batch_size=self.batch_size, block_size=self.block_size)
    
    def load_data(self, path, train_frac=0.9):
        with open(path, encoding="utf-8") as f:
            text = "".join(c for c in f.read() if c in PERMITED)
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(train_frac * len(data))
        return data[:n], data[n:], chars


    def batches(self, data, n_batches, batch_size, block_size):

        if len(data) < batch_size * block_size:
            raise ValueError(f"data length {len(data)} is too small for batch_size {batch_size} and block_size {block_size}")
        
        for _ in range(n_batches):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i + block_size] for i in ix])
            y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
            yield x, y

def train(model):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = DataLoader(n_batches=100, batch_size=32, block_size=model.block_size)
    for x, y in dataloader:
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():.4f}")

model = Decoder_Only_Transformer(vocab_size=100, block_size=32, d_model=64, n_heads=4, d_k=16, n_layers=8)
train(model)