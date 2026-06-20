import torch
import torch.nn as nn
from torch.nn import functional as F


BATCH_SIZE = 4
BLOCK_SIZE = 8
LEARNING_RATE = 1e-2
MAX_ITERS = 1000
EVAL_INTERVAL = 100
EVAL_ITERS = 20


def load_text():
    # Read the raw training text from input.txt.
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text


def build_vocab(text):
    # Build character vocabulary plus encode/decode helpers.
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    return chars, vocab_size, encode, decode


def make_data(text, encode):
    # Convert the full text into token ids and split train/validation data.
    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(split, train_data, val_data):
    # Sample random context windows x and next-token targets y.
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    return x, y


class BigramLanguageModel(nn.Module):
    # Predict next-token logits directly from the current token id.
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    # Evaluate average train and validation loss without updating weights.
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


def train(model, train_data, val_data):
    # Run the optimization loop over sampled batches.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = get_batch("train", train_data, val_data)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def generate_sample(model, decode):
    # Generate text autoregressively from the trained model.
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=300)
    print(decode(generated[0].tolist()))


def main():
    # Wire the data, model, training loop, and generation together.
    text = load_text()
    chars, vocab_size, encode, decode = build_vocab(text)
    train_data, val_data = make_data(text, encode)

    print(f"vocab size: {vocab_size}")
    print(f"train tokens: {len(train_data)}")
    print(f"val tokens: {len(val_data)}")
    
    x, y = get_batch("train", train_data, val_data)
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    model = BigramLanguageModel(vocab_size)

    train(model, train_data, val_data)

    generate_sample(model, decode)


if __name__ == "__main__":
    main()
