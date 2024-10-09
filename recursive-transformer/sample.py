import torch
import os
from model import *
import tiktoken

torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model = GPT(GPTConfig(vocab_size=50304))
  model.to(device)
  save_dir = 'outputs'

  load_path = os.path.join(save_dir, "model_final.pt")

  if os.path.exists(load_path):
      checkpoint = torch.load(load_path, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      print(f"Model loaded from {load_path}")
  else:
      print(f"No saved model found at {load_path}")

  model.eval()
  num_return_sequences = 5
  max_length = 30
  enc = tiktoken.get_encoding('gpt2')
  tokens = enc.encode("Hello, I'm a language model,")
  tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
  tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
  x = tokens.to(device)
  while x.size(1) < max_length:
    with torch.no_grad():
      logits, _ = model(x) # (B, T, vocab_size)

      logits = logits[:, -1, :] # (B, vocab_size)
      probs = F.softmax(logits, dim=-1)
      topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
      ix = torch.multinomial(topk_probs, 1) # (B, 1)
      xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
      x = torch.cat((x, xcol), dim=1)
  # print the generated text
  for i in range(num_return_sequences):
      tokens = x[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(">", decoded)
