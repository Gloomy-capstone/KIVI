"""KV cache fake quantization (round-to-nearest, asymmetric, group-wise).

KIVI 논문(Liu et al., 2024) Section 3.1의 fake quantization 정의를 따른다.
- per-token : token마다 channel(D)을 group_size 단위로 묶어 양자화
- per-channel: channel마다 token(T)을 group_size 단위로 묶어 양자화
                 token 수가 group_size로 나누어 떨어지지 않으면 0으로 padding

여기서 'fake'란 실제로 저정밀로 저장하지 않고, fp16에서 Q→DQ 한 값을 그대로
attention 계산에 사용한다는 뜻. 즉, 정확도 영향만 평가한다.
"""

import torch


def _quant_dequant_along_last(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Asymmetric RTN. x의 마지막 차원을 한 그룹으로 보고 Q→DQ.

        z = min(x), s = (max(x) - min(x)) / (2^B - 1)
        Q(x) = round((x - z) / s),   x' = Q(x) * s + z
    """
    qmax = (1 << n_bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    x_q = torch.round((x - x_min) / scale).clamp_(0, qmax)
    return x_q * scale + x_min


def fake_quant_per_token(x: torch.Tensor, n_bits: int, group_size: int) -> torch.Tensor:
    """Per-token 양자화. shape: [B, H, T, D].

    채널 차원 D를 group_size 단위로 묶고, 각 (b, h, t, group)에 대해 독립적으로
    scale/zero-point를 계산한다. 한 token 안에서 양자화 오차가 갇힌다.
    """
    B, H, T, D = x.shape
    assert D % group_size == 0, (
        f"head_dim {D}이 group_size {group_size}로 나누어떨어져야 합니다. "
        f"per-token quantization."
    )
    # [B, H, T, D] -> [B, H, T, D//G, G]
    x_g = x.reshape(B, H, T, D // group_size, group_size)
    x_dq = _quant_dequant_along_last(x_g, n_bits)
    return x_dq.reshape(B, H, T, D)


def fake_quant_per_channel(x: torch.Tensor, n_bits: int, group_size: int) -> torch.Tensor:
    """Per-channel 양자화. shape: [B, H, T, D].

    token 차원 T를 group_size 단위로 묶고, 각 (b, h, t_group, d)에 대해 독립적으로
    scale/zero-point를 계산한다. 한 channel 안에서 양자화 오차가 갇힌다.
    T가 group_size의 배수가 아니면 zero-padding 후 양자화하고 다시 잘라낸다.
    (논문 Section 3.1, "we add zero-padding to ensure it can be grouped perfectly.")
    """
    B, H, T, D = x.shape
    pad = (group_size - T % group_size) % group_size
    if pad > 0:
        zeros = torch.zeros(B, H, pad, D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, zeros], dim=2)
    T_p = x.shape[2]

    # [B, H, T_p, D] -> [B, H, T_p//G, G, D]
    # 그룹 차원(G)이 마지막에서 두 번째이므로, 마지막 두 차원을 swap해서
    # _quant_dequant_along_last를 그대로 재사용한다.
    x_g = x.reshape(B, H, T_p // group_size, group_size, D)
    x_g = x_g.transpose(-1, -2).contiguous()  # [B, H, T_p//G, D, G]
    x_dq = _quant_dequant_along_last(x_g, n_bits)
    x_dq = x_dq.transpose(-1, -2).contiguous()  # [B, H, T_p//G, G, D]

    x_dq = x_dq.reshape(B, H, T_p, D)
    if pad > 0:
        x_dq = x_dq[:, :, :T, :]
    return x_dq


def apply_kv_fake_quant(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    k_bits: int,
    v_bits: int,
    k_dim: str,
    v_dim: str,
    group_size: int,
):
    """K, V cache에 fake quantization을 적용해 dequantized 텐서를 반환.

    bits >= 16이면 해당 캐시는 양자화하지 않는다(baseline 용).
    """
    if k_bits < 16:
        if k_dim == "token":
            key_states = fake_quant_per_token(key_states, k_bits, group_size)
        elif k_dim == "channel":
            key_states = fake_quant_per_channel(key_states, k_bits, group_size)
        else:
            raise ValueError(f"Unknown k_dim: {k_dim!r} (use 'token' or 'channel')")

    if v_bits < 16:
        if v_dim == "token":
            value_states = fake_quant_per_token(value_states, v_bits, group_size)
        elif v_dim == "channel":
            value_states = fake_quant_per_channel(value_states, v_bits, group_size)
        else:
            raise ValueError(f"Unknown v_dim: {v_dim!r} (use 'token' or 'channel')")

    return key_states, value_states
