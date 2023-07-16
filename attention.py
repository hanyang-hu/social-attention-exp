import torch
import torch.nn.functional as F
import math


'''Multi-head scaled dot-product ego-oppo-attention'''
class Attention(torch.nn.Module):
    def __init__(self, ego_dim, oppo_dim, embed_dim = 96, num_heads = 3):
        super(Attention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = torch.nn.Linear(ego_dim, embed_dim)
        self.kv_proj = torch.nn.Linear(oppo_dim, 2*embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialization of parameters
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def forward(self, ego, oppo):
        # Accept both batched and unbatched input
        is_batched = ego.dim() > 1
        if not is_batched:
            ego = ego.unsqueeze(0)
            oppo = oppo.unsqueeze(0)
        batch_size = ego.size(0)

        # Compute and separate Q, K, V from linear output
        q = self.q_proj(ego).reshape(batch_size, self.num_heads, 1, self.head_dim)
        k, v = self.kv_proj(oppo).reshape(batch_size, oppo.size()[1], self.num_heads, 2*self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=-1)

        # Determine value and attention outputs
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # broadcasting
        attn_logits = attn_logits / math.sqrt(self.head_dim) # d_k == head_dim
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v).reshape(batch_size, self.embed_dim)
        o = self.o_proj(values)

        return o if is_batched else o.squeeze(0)
