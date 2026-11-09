import argparse
import time
import torch
from tokenizers import Tokenizer
from model import Decoder_Only_Transformer
from train import PERMITTED

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="lorca_char_DecoderOnlyTransformer.pt")
parser.add_argument("--prompt", default="\n")
parser.add_argument("--tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=20)
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location=device)
model = DecoderOnlyTransformerOpt(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

if "chars" in ckpt:
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = set(chars)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)
else:
    tok = Tokenizer.from_str(ckpt["tokenizer"])
    vocab = PERMITTED
    encode = lambda s: tok.encode(s).ids
    decode = tok.decode

prompt = "".join(c for c in args.prompt if c in vocab) or "\n"
ids = encode(prompt)
idx = torch.tensor([ids], dtype=torch.long, device=device)

text = prompt
print(text, end="", flush=True)
for next_id in model.generate_stream(idx, args.tokens, temperature=args.temperature, top_k=args.top_k):
    ids.append(next_id)
    new_text = decode(ids)          # decode everything: a BPE token can be half of an "á"
    if new_text.endswith("\ufffd"):  # incomplete character, wait for the next byte
        continue
    print(new_text[len(text):], end="", flush=True)
    text = new_text
    time.sleep(0.02)