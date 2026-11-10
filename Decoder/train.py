import os
import json
import time
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from model import DecoderOnlyTransformer
from model_optimized import DecoderOnlyTransformerOpt


'''''I would like to remark some optimizations that have been made in the model_optimized.py file:
    - moving the model to the apple silicon GPU
    - compiling the model with torch.compile, which is a new feature in PyTorch 2.0
    - compiling the training and evaluation graphs outside the timer to avoid including compilation time in the training time measurement
    - using torch.autocast to automatically cast tensors to the appropriate data type for faster computation
    - using torch.no_grad to disable gradient calculation during evaluation, which can save memory and computation
    - vectorization of the funcion get_batches and instantiation of the data tensor in the GPU
    - fused calculations in the AdamW optimizer, reduction of kernel launches
    '''

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

    offsets = torch.arange(block_size + 1, device=data.device) #(T + 1,) tensor
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device).unsqueeze(1)  # (B, 1)
        chunk = data[ix + offsets]  # (B, block_size+1), single gather

        yield chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data, batch_size, iters=20):

    model.eval()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        losses = [model(x, y)[1] for x, y in batches(data, iters, batch_size, model.block_size)]
    model.train()

    return torch.stack(losses).mean().item()


def train(model, train_data, val_data, steps, batch_size, lr=1e-3, eval_every=100):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    history = []

    for step, (x, y) in enumerate(batches(train_data, steps, batch_size, model.block_size)):
        if step % eval_every == 0:
            train_loss = evaluate(model, train_data, batch_size)
            val_loss = evaluate(model, val_data, batch_size)
            history.append((step, train_loss, val_loss, ))
            print(f"step {step}: "f"train loss {train_loss:.4f}, "f"val loss {val_loss:.4f}, "f"tokens/s {tokens_per_s:.4f}")

        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        torch.mps.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        tokens_per_s = batch_size * model.block_size / elapsed

    return history

transformers = [DecoderOnlyTransformerOpt, DecoderOnlyTransformer]

if __name__ == "__main__":

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    text = load_text(PATH)
    n = int(0.9 * len(text))
    train_text, val_text = text[:n], text[n:]
    train_cfg = dict(steps=2000, batch_size=32, lr=1e-3, eval_every=100)
    os.makedirs("logs", exist_ok=True)

    for transformer in transformers:

        for name, bpe_vocab in [("char", None), ("bpe-512", 512)]:

            train_data, val_data, vocab_size, tok_state = tokenize(train_text, val_text, bpe_vocab)
            train_data = train_data.to(device)
            val_data = val_data.to(device)

            config = dict(vocab_size=vocab_size, block_size=32, d_model=12, n_heads=4, n_layers=1)
            model = transformer(**config).to(device)
            model = torch.compile(model, mode="reduce-overhead")

            # warm-up: compile train and eval graphs outside the timer
            x, y = next(batches(train_data, 1, train_cfg["batch_size"], model.block_size))
            with torch.autocast(device_type=device, dtype=torch.bfloat16):   #we are training in bfloat16, the warm-up has to be done in the same precision
                model(x, y)[1].backward()
            model.zero_grad(set_to_none=True)
            evaluate(model, val_data, train_cfg["batch_size"], iters=1)

            torch.mps.synchronize()
            t0 = time.time()
            history = train(model, train_data, val_data, **train_cfg)
            torch.mps.synchronize()
            elapsed = time.time() - t0
            tokens_per_s_mean = (train_cfg["steps"] * train_cfg["batch_size"] * model.block_size) / elapsed
            r_train = len(train_data) / len(train_text)
            r_val = len(val_data) / len(val_text)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log = {
                "name": name,
                "date": stamp,
                "device": device,
                "model": config,
                "n_params": sum(p.numel() for p in model.parameters()),
                "train": train_cfg,
                "tokens": dict(train=len(train_data), val=len(val_data), chars_per_token=1 / r_train),
                "train_time_s": elapsed,
                "tokens_per_s_mean": tokens_per_s_mean,
                "history": [dict(step=s, train=t, val=v, train_per_char=t * r_train, val_per_char=v * r_val)
                            for s, t, v in history],
            }
            with open(f"logs/{stamp}_{name}.json", "w") as f:
                json.dump(log, f, indent=2)
            torch.save({"model": model.state_dict(), "config": config, **tok_state}, f"lorca_{name}_{transformer.__name__}.pt")
