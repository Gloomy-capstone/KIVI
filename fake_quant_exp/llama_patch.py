"""LlamaAttention.forward를 fake-quantization 버전으로 monkey-patch.

transformers 4.36.2의 표준 LlamaAttention(eager) 구현을 그대로 따르되,
KV cache concat 직후 — repeat_kv 직전에 — fake quantization을 끼워 넣는다.
이렇게 하면 prefill에서도, decoding의 매 step에서도 항상 'KV cache 전체'가
저정밀 형태로 attention에 사용되는 것과 동치가 된다.

NOTE
----
- transformers의 SdpaAttention / FlashAttention2는 patch하지 않으므로,
  반드시 모델을 ``attn_implementation="eager"``로 로드해야 한다.
- patch_llama()는 모델 로딩 전에 호출해도 되고 후에 호출해도 된다.
  클래스 메서드 자체를 교체하기 때문에 모든 instance에 즉시 반영된다.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from fake_quant import apply_kv_fake_quant


# 전역 config — run_eval.py에서 매 실험 전에 set_kv_quant_config로 갱신한다.
_CFG = {
    "k_bits": 16,
    "v_bits": 16,
    "k_dim": "token",
    "v_dim": "token",
    "group_size": 32,
}


def set_kv_quant_config(k_bits, v_bits, k_dim, v_dim, group_size=32):
    _CFG["k_bits"] = int(k_bits)
    _CFG["v_bits"] = int(v_bits)
    _CFG["k_dim"] = k_dim
    _CFG["v_dim"] = v_dim
    _CFG["group_size"] = int(group_size)
    print(
        f"[fake-quant] K={_CFG['k_bits']}bit/{_CFG['k_dim']}, "
        f"V={_CFG['v_bits']}bit/{_CFG['v_dim']}, group={_CFG['group_size']}"
    )


def fake_quant_llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ===== Fake quantization on the FULL KV cache =====
    # 캐시에 저장된 fp16 K, V를 통째로 Q→DQ.
    # per-channel은 매 step마다 token 수가 달라지므로 zero-padding이 적용된다.
    key_states, value_states = apply_kv_fake_quant(
        key_states,
        value_states,
        k_bits=_CFG["k_bits"],
        v_bits=_CFG["v_bits"],
        k_dim=_CFG["k_dim"],
        v_dim=_CFG["v_dim"],
        group_size=_CFG["group_size"],
    )
    # ==================================================

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # softmax in fp32 for numerical stability (Llama 표준 구현 그대로)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def patch_llama():
    """LlamaAttention.forward를 fake-quant 버전으로 교체."""
    LlamaAttention.forward = fake_quant_llama_attn_forward
    print("[fake-quant] LlamaAttention.forward has been patched.")
