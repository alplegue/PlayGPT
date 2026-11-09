import torch
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from model import Decoder_Only_Transformer

PATH = "Obras_completas_ilustradas_Federico_Garcia_Lorca_djvu.txt"
PERMITTED = set("abcdefghijklmnñopqrstuvwxyzáéíóúü"
               "ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ"
               " \n.,;:¡!¿?-'\"()")

def load_text(path):
    with open(path, encoding="utf-8") as f:
        return "".join(c for c in f.read() if c in PERMITTED)


def BPETokenizer_train(text, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text], trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def tokenize(train_text, val_text, bpe_vocab=None):
    if bpe_vocab is None:
        chars = sorted(set(train_text + val_text))
        stoi = {c: i for i, c in enumerate(chars)}
        enc, vocab_size, tok_state = (lambda s: [stoi[c] for c in s]), len(chars), {"chars": chars}
    else:
        tok = BPETokenizer_train(train_text, bpe_vocab)
        enc, vocab_size, tok_state = (lambda s: tok.encode(s).ids), tok.get_vocab_size(), {"tokenizer": tok.to_str()}

    return torch.tensor(enc(train_text)), torch.tensor(enc(val_text)), vocab_size, tok_state

def batches(data, n_batches, batch_size, block_size):
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        yield x, y


@torch.no_grad()
def evaluate(model, data, batch_size, iters=20):
    model.eval()
    losses = [model(x, y)[1] for x, y in batches(data, iters, batch_size, model.block_size)]
    model.train()
    return torch.stack(losses).mean().item()


def train(model, train_data, val_data, steps, batch_size, eval_every=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []

    for step, (x, y) in enumerate(batches(train_data, steps, batch_size, model.block_size)):
        if step % eval_every == 0:
            train_loss = evaluate(model, train_data, batch_size)
            val_loss = evaluate(model, val_data, batch_size)
            history.append((train_loss, val_loss))
            print(f"step {step}: "f"train loss {train_loss:.4f}, "f"val loss {val_loss:.4f}")

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history


if __name__ == "__main__":

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    text = load_text(PATH)
    n = int(0.9 * len(text))
    train_text, val_text = text[:n], text[n:]
    results = {}

    for name, bpe_vocab in [("char", None), ("bpe-512", 512)]:

        train_data, val_data, vocab_size, tok_state = tokenize(train_text, val_text, bpe_vocab)
        train_data = train_data.to(device)
        val_data = val_data.to(device)

        config = dict(vocab_size=vocab_size, block_size=64, d_model=64, n_heads=4, n_layers=4)
        model = Decoder_Only_Transformer(**config).to(device)
        history = train(model, train_data, val_data, steps=2000, batch_size=32, eval_every=100)
        r_train = len(train_data) / len(train_text)
        r_val = len(val_data) / len(val_text)
        results[name] = [(t * r_train, v * r_val) for t, v in history]
        torch.save({"model": model.state_dict(), "config": config, **tok_state}, f"lorca_{name}.pt")
