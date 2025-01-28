"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from longformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b



class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(CausalSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        return outputs


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)




@dataclass
class GPTConfig:
    attn_mode: str = 'local'
    local_attn_ctx: int = 32 * 4
    
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    


