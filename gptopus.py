# GPTopus

# gpt: generatively pre-trained transformer
# decoder-only model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken

''' Check Device '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' Hyperparameters '''
batch_size = 32 # number of independent sequences processed in parallel
block_size = 64 # max context length for predictions
eval_itrs = 200
max_itrs = 4000
eval_intrvl = 200
learning_rate = 2e-4
max_new_tkns = 1000 # generated text length
n_embd = 192 # embedding dimension
n_heads = 6
n_layers = 6
dropout = 0.2

''' Dataset '''
filename = 'input.txt' # set as input text
with open(filename, 'r', encoding='utf-8') as file:
  text = file.read()

''' Get Unique chars '''
chars = sorted(list(set(text)))

'''
Get Tokeniser for Mapping (chars to int)
token: sub pieces
encoder: takes string outputs list of int
decoder: takes list of int outputs string
'''

# get tokeniser for model in OpenAI API
enc = tiktoken.encoding_for_model("gpt-4o") # change gpt model
vocab_size = enc.n_vocab # vocab_size

''' Encode Data '''
data = torch.tensor(enc.encode(text), dtype=torch.long)

''' Split Data '''
split = int(0.9 * len(data))
data_train = data[:split]
data_val = data[split:]

''' Create Batch of Data w/ Inputs: x, Targets: y '''
def get_batch(split):
  data = data_train if split == 'train' else data_val
  # make tensor for random starting index of block for each batch
  last_idx = len(data) - block_size
  rand_idx = torch.randint(last_idx, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in rand_idx]) # input tensor (can include entire block)
  y = torch.stack([data[i+1:i+block_size+1] for i in rand_idx]) # target tensor (after input)
  x, y = x.to(device), y.to(device) # send to device
  return x, y

''' Calculate Loss '''
# do not store gradients in memory
@torch.no_grad()
def calculate_loss():
  avg_loss = {}
  # evaluation mode (no dropout)
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_itrs)
    for v in range(eval_itrs):
      x, y = get_batch(split)
      _, loss = model(x, y)
      losses[v] = loss.item() # get loss value as float
    avg_loss[split] = losses.mean() # get average loss for batch
  # training mode (dropout)
  model.train()
  return avg_loss


class Head(nn.Module):
  ''' 1 self-attention head '''

  def __init__(self, head_size):
    super().__init__()
    self.key = self.query = self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # create buffer tensor
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    '''
    B: batch
    T: time
    C: channel

    wt: weights
    '''
    B, T, C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)
    v = self.value(x) # (B, T, C)
    # affinities: attention scores
    # calculate affinities
    wt = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
    wt = wt.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wt = F.softmax(wt, dim=-1) # (B, T, T)
    wt = self.dropout(wt)
    # weighted aggregation of values
    wt_agg = wt @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    return wt_agg


class MultiHeadAttention(nn.Module):
  ''' multiple self-attention heads in parallel '''

  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    sattn_out = torch.cat([h(x) for h in self.heads], dim=-1) # concat over channel dim
    sattn_out = self.proj(sattn_out) # apply projection on self-attention output
    sattn_out = self.dropout(sattn_out)
    return sattn_out
 

class FeedForward(nn.Module):
  ''' feed forward: linear layer followed by non-linearity '''

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), # feedforward channel sizes multiplied by 4
        nn.ReLU(), # non-linear
        nn.Linear(4 * n_embd, n_embd), # projection
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)
  

''' 
layer normalization: compute mean and variance to normalize all summed inputs to neuron layer for single training case and give each neuron an adaptive bias and gain, reduces training time

y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
'''


class TransformerBlock(nn.Module):
  ''' transformer block: node communication followed by computation '''

  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sattn = MultiHeadAttention(n_heads, head_size)
    self.ffwd = FeedForward(n_embd)
    self.lnorm1 = nn.LayerNorm(n_embd)
    self.lnorm2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # note: cannot use += due to gradient computation
    x = x + self.sattn(self.lnorm1(x)) # add self-attention to computation
    x = x + self.ffwd(self.lnorm2(x)) # add feedforward to computation
    return x


'''
Bigram Language Model: statistical model that predicts likelihood of next char given previous chars
'''


class BigramLangModel(nn.Module):
  ''' Bigram Language Model '''

  def __init__(self):
    super().__init__()
    # token reads logits for next token from table
    self.tkn_embd_table = nn.Embedding(vocab_size, n_embd) # token embedding
    self.pos_embd_table = nn.Embedding(block_size, n_embd) # position embedding
    self.blocks = nn.Sequential(
        *[TransformerBlock(n_embd, n_heads=n_heads) for _ in range(n_layers)],
    )
    self.lnorm_end = nn.LayerNorm(n_embd) # layer norm @ end
    self.lm_head = nn.Linear(n_embd, vocab_size) # language model head

  def forward(self, idx, targets=None):
    '''
    idx: (B, T)
    targets: (B, T)

    both are tensors of integers
    '''
    B, T = idx.shape
    tkn_embd = self.tkn_embd_table(idx) # (B, T, C)
    pos_embd = self.pos_embd_table(torch.arange(T, device=device)) # (T, C)
    x = tkn_embd + pos_embd # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    x = self.lnorm_end(x) # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      # reshape for cross-entropy input
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tkns):
    '''
    generate new tokens

    idx: (B, T)
    '''
    for _ in range(max_new_tkns):
      last_idx = idx[:, -block_size:] # last block size token
      logits, _ = self(last_idx) # get predictions
      logits = logits[:, -1, :] # (B, C) use last time step
      probs = F.softmax(logits, dim=-1) # (B, C) get probabilities
      next_idx = torch.multinomial(probs, num_samples=1) # (B, 1) sample distribution
      idx = torch.cat((idx, next_idx), dim=1) # (B, T+1) append sample idx
    return idx
  

''' Model '''
model = BigramLangModel()
m = model.to(device) # send model to device


''' Optimizer '''
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for itr in range(max_itrs):
  # evaluate loss @ interval
  if itr % eval_intrvl == 0:
    losses = calculate_loss()
    print(f"step: {itr}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

  # get batch sample
  x_train, y_train = get_batch('train')

  # calculate loss
  _, loss = model(x_train, y_train)
  optimizer.zero_grad(set_to_none=True) # clear gradients
  loss.backward() # calculate gradients dloss/dx for every parameter x
  optimizer.step() # update optimizer


''' Generate from Model '''
prev_idx = torch.zeros((1, 1), dtype=torch.long, device=device) # previous token index
gen_text = enc.decode(m.generate(prev_idx, max_new_tkns=max_new_tkns)[0].tolist()) # generate new text

# put generated text in new file
with open(filename.replace('.txt', '_out.txt'), 'w') as file:
  file.write(gen_text)