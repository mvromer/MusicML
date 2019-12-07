import torch
from musicml import *

h = Hyperparameters( vocab_size=5 )
mt = MusicTransformer( h )
S = 2
T = 4
inp = torch.randint( h.vocab_size, (S,) )
out = torch.randint( h.vocab_size, (T,) )
mask = create_attention_mask( T, T )
z = mt( inp, out, mask )
