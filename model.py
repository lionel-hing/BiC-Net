import numpy as np
import torch
from torch import nn
from transformers import AutoModel
import utils


class BiC(nn.Module):
    """BiC-Net model"""

    def __init__(self, args):
        super(BiC, self).__init__()

        self.global_dim = args.global_dim
        self.region_dim = args.region_dim
        self.text_dim = args.text_dim
        self.num_heads = args.num_heads
        self.region_num = args.region_num
        self.embedding_dim = args.embedding_dim
        self.layer_num = args.layer_num
        self.dropout = args.dropout
        self.mlp_ration = args.mlp_ration
        self.attn_drop = args.attn_drop

        # region video
        self.proj_region = nn.Linear(self.region_dim, self.embedding_dim)
        self.norm_region = LayerNormalization(self.embedding_dim)
        self.embedding_region = PositionalEncoding(self.embedding_dim, self.dropout)

        # spatial temporal transformer
        self.S_T_trans_region = nn.ModuleList(
            [TransformerEncoder(2, self.embedding_dim, self.num_heads,
                                int(self.embedding_dim * self.mlp_ration), self.attn_drop)
             for _ in range(self.layer_num)])
        self.S_pool_region = nn.AvgPool2d((self.region_num, 1))
        self.T_pool_region = AtnPool(self.embedding_dim, self.embedding_dim * 2, 2, self.dropout)
        self.norm = LayerNormalization(self.embedding_dim)
        self.norm_1 = LayerNormalization(self.embedding_dim)
        self.dropout_region = nn.Dropout(0.1)

        # global video
        self.proj_global = nn.Linear(self.global_dim, self.embedding_dim)
        self.norm_global = LayerNormalization(self.embedding_dim)
        self.embedding_global = PositionalEncoding(self.embedding_dim, self.dropout)
        self.T_Trans_global = nn.ModuleList(
            [TransformerEncoder(1, self.embedding_dim, self.num_heads,
                                int(self.embedding_dim * self.mlp_ration), self.attn_drop)
             for _ in range(self.layer_num)])
        self.T_pool_global = AtnPool(self.embedding_dim, self.embedding_dim * 2, 2, self.dropout)
        self.norm_2 = LayerNormalization(self.embedding_dim)
        self.dropout_global = nn.Dropout(0.1)

        # text
        self.norm_text = LayerNormalization(self.embedding_dim)
        self.proj_text = nn.Linear(self.text_dim, self.embedding_dim)
        self.pool_text = AtnPool(self.embedding_dim, self.embedding_dim * 2, 2, self.dropout)
        self.norm_3 = LayerNormalization(self.embedding_dim)
        self.dropout_text = nn.Dropout(0.1)


        init_network(self, 0.01)

    def forward(self, text_feats, region_feats, global_feats, text_mask, region_mask, global_mask):

        # relation embedding
        region_emb = self.norm_region(self.proj_region(region_feats))
        batch_size, seq_len, region_num, region_dim = region_emb.shape
        region_emb = region_emb.reshape((-1, region_num, region_dim))
        region_emb = self.embedding_region(region_emb)
        region_emb = region_emb.reshape((batch_size, seq_len, region_num, region_dim))

        # mask
        single_region_mask = region_mask[0]
        batch_size, query_len, embed_dim = region_emb[0].shape
        batch_size, key_len, embed_dim = region_emb[0].shape
        single_region_mask = (1 - single_region_mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        single_region_mask = single_region_mask == 1

        batch_size, query_len, region_len, embed_dim = region_emb.shape
        batch_size, key_len, region_len, embed_dim = region_emb.shape
        region_mask = region_mask.reshape((batch_size * query_len, region_len))
        region_mask = (1 - region_mask.unsqueeze(1).expand(batch_size * query_len, region_len, region_len))
        region_mask = region_mask == 1
        for layer in self.S_T_trans_region:
            region_emb_raw = region_emb
            # spatial
            region_emb_output = torch.zeros_like(region_emb)
            for i in range(batch_size):
                single_region_emb = region_emb[i]
                frame_region_emb = layer.encoder_layers[0](region_emb[i], region_emb[i], region_emb[i],
                                                           single_region_mask)
                region_emb_output[i] = frame_region_emb
          
            region_emb_output = self.norm(region_emb_output + region_emb_raw)
           
            region_emb_final = layer.encoder_layers[1](region_emb_output, region_emb_output, region_emb_output, region_mask)
            
            region_emb_final = self.norm(region_emb_final + region_emb_output)
            
            region_emb = self.norm(region_emb_final + region_emb_raw)
        region_emb = (self.S_pool_region(region_emb)).squeeze(2)
        region_emb = self.T_pool_region(region_emb, global_mask)
        region_emb = self.norm_1(region_emb)

        # global video embedding
        global_emb = self.norm_global(self.proj_global(global_feats))
        global_emb = self.embedding_global(global_emb)
        for encoder_layer in self.T_Trans_global:
            #global_emb_raw = global_emb
            global_emb = encoder_layer(global_emb, global_emb, global_emb, global_mask)
            #global_emb = self.norm_2(global_emb + global_emb_raw)
        global_emb = self.T_pool_global(global_emb, global_mask)
        global_emb = self.norm_2(global_emb)

        # text embedding
        text_emb = self.norm_text(self.proj_text(text_feats))
        text_emb = self.pool_text(text_emb, text_mask)
        text_emb = self.norm_3(text_emb)

        return text_emb, region_emb, global_emb


class LayerNormalization(nn.Module):
    """Layer Normalization"""
    def __init__(self, normalized_shape):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.epsilon = 1e-06

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class PositionalEncoding(nn.Module):
    """normal sinusoidalpostion embedding"""
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads_count, d_ff, dropout_prob)
                for _ in range(layers_count)])

        init_network(self, 0.01)

    def forward(self, query, key, value, mask):
        if query.ndim == 3:
            batch_size, query_len, embed_dim = query.shape
            batch_size, key_len, embed_dim = key.shape
            mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        else:
            batch_size, query_len, region_len, embed_dim = query.shape
            batch_size, key_len, region_len, embed_dim = key.shape
            mask = mask.reshape((batch_size * query_len, region_len))
            mask = (1 - mask.unsqueeze(1).expand(batch_size * query_len, region_len, region_len))
        mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)

        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedfoward_layer = Sublayer(
            PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, qeury, key, value, sources_mask):
        sources = self.self_attention_layer(qeury, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedfoward_layer(sources)

        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0, \
            f"model dim {d_model} not divisible by {heads_count} heads"
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        if query.ndim == 3:
            batch_size, query_len, d_model = query.size()
        else:
            batch_size, query_len, region_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)

        if key.ndim == 3:
            batch_size, key_len, d_model = key_projected.size()
            batch_size, value_len, d_model = value_projected.size()
            query_heads = query_projected.view(
                batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
            key_heads = key_projected.view(
                batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
            value_heads = value_projected.view(
                batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        else:
            batch_size, key_len, region_len, d_model = key_projected.size()
            key_projected.reshape((-1, region_len, d_model))
            batch_size, value_len, region_len, d_model = value_projected.size()
            value_projected.reshape((-1, region_len, d_model))
            query_projected.reshape((-1, region_len, d_model))
            query_heads = query_projected.view(
                batch_size * query_len, region_len, self.heads_count, d_head).transpose(1, 2)
            key_heads = key_projected.view(
                batch_size * key_len, region_len, self.heads_count, d_head).transpose(1, 2)
            value_heads = value_projected.view(
                batch_size * value_len, region_len, self.heads_count, d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        if query.ndim == 3:
            context = context_sequence.reshape(batch_size, query_len, d_model)
        else:
            context = context_sequence.reshape(batch_size, query_len, region_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)

        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    """MLP layer"""
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)


class AtnPool(nn.Module):
    """Attention-aware feature aggregation layer"""
    def __init__(self, d_input, d_attn, n_heads, dropout_prob):
        super(AtnPool, self).__init__()
        self.d_head = d_attn // n_heads
        self.d_head_output = d_input // n_heads
        self.num_heads = n_heads

        def init_(tensor_):
            tensor_.data = (utils.truncated_normal_fill(tensor_.data.shape, std=0.01))

        w1_head = torch.zeros(n_heads, d_input, self.d_head)
        b1_head = torch.zeros(n_heads, self.d_head)
        w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
        b2_head = torch.zeros(n_heads, self.d_head_output)
        init_(w1_head)
        init_(b1_head)
        init_(w2_head)
        init_(b2_head)
        self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
        self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
        self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
        self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_temp = 1
        self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=True)

    def extra_repr(self) -> str:
        strs = []
        for p in [self.genpool_w1_head, self.genpool_b1_head,
                  self.genpool_w2_head, self.genpool_b2_head]:
            strs.append(f"pool linear {p.shape}")
        return "\n".join(strs)

    def forward(self, features, mask):
        if features.ndim == 3:
            batch_size, seq_len, input_dim = features.shape
            b1 = torch.matmul(features.unsqueeze(1), self.genpool_w1_head.unsqueeze(0))
            b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
            b1 = self.activation(self.dropout1(b1))
            b1 = torch.matmul(b1, self.genpool_w2_head.unsqueeze(0))
            b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
            b1 = self.dropout2(b1)
            b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)
        else:
            batch_size, seq_len, region_num, input_dim = features.shape
            b1 = torch.matmul(features.unsqueeze(1), self.genpool_w1_head.unsqueeze(1).unsqueeze(0))
            b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(1).unsqueeze(0)
            b1 = self.activation(self.dropout1(b1))
            b1 = torch.matmul(b1, self.genpool_w2_head.unsqueeze(1).unsqueeze(0))
            b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(1).unsqueeze(0)
            b1 = self.dropout2(b1)
            b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)
        smweights = self.softmax(b1 / self.softmax_temp)
        smweights = self.dropout3(smweights)
        if features.ndim == 3:
            smweights = smweights.transpose(1, 2).reshape(-1, seq_len, input_dim)
        else:
            smweights = smweights.transpose(1, 3).reshape(-1, seq_len, region_num, input_dim)
        pooled = (features * smweights).sum(dim=-2)
        return pooled


def init_weight_(w, init_gain=1):
    w.copy_(utils.truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):
    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            init_weight_(val.data, init_std)
