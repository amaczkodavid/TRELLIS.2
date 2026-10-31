from typing import List, Union, overload
import torch
import torch.nn.functional as F
from .. import VarLenTensor
from .. import config

__all__ = [
    'sparse_scaled_dot_product_attention',
]


def _varlen_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: List[int],
    kv_seqlen: List[int],
) -> torch.Tensor:
    if len(q_seqlen) != len(kv_seqlen):
        raise ValueError(
            f'Number of query and key/value sequences must match, got {len(q_seqlen)} and {len(kv_seqlen)}'
        )
    if q.shape[0] != sum(q_seqlen):
        raise ValueError(
            f'Query tensor holds {q.shape[0]} tokens but the sequence lengths sum to {sum(q_seqlen)}'
        )
    if k.shape[0] != sum(kv_seqlen):
        raise ValueError(
            f'Key tensor holds {k.shape[0]} tokens but the sequence lengths sum to {sum(kv_seqlen)}'
        )
    if v.shape[0] != sum(kv_seqlen):
        raise ValueError(
            f'Value tensor holds {v.shape[0]} tokens but the sequence lengths sum to {sum(kv_seqlen)}'
        )
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f'Expected 3D query, key, and value tensors, got {q.dim()}D, {k.dim()}D, and {v.dim()}D'
        )
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(
            f'Head count mismatch, got {q.shape[1]}, {k.shape[1]}, and {v.shape[1]}'
        )
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f'Query and key embedding dimension mismatch, got {q.shape[2]} and {k.shape[2]}'
        )

    num_heads = q.shape[1]
    head_dim_v = v.shape[2]
    outputs = []
    q_start = 0
    kv_start = 0

    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        q_len = int(q_len)
        kv_len = int(kv_len)

        if q_len == 0:
            outputs.append(q.new_empty((0, num_heads, head_dim_v)))
            kv_start += kv_len
            continue

        if kv_len == 0:
            outputs.append(q.new_zeros((q_len, num_heads, head_dim_v)))
            q_start += q_len
            continue

        q_i = q[q_start:q_start + q_len].transpose(0, 1).unsqueeze(0)
        k_i = k[kv_start:kv_start + kv_len].transpose(0, 1).unsqueeze(0)
        v_i = v[kv_start:kv_start + kv_len].transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i)
        outputs.append(out_i[0].transpose(0, 1).reshape(q_len, num_heads, head_dim_v))

        q_start += q_len
        kv_start += kv_len

    if q_start != q.shape[0] or kv_start != k.shape[0]:
        raise ValueError('Attention sequence bookkeeping error')

    if len(outputs) == 0:
        return q.new_empty((0, num_heads, head_dim_v))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: VarLenTensor,
    kv: Union[VarLenTensor, torch.Tensor],
) -> VarLenTensor:
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    kv: VarLenTensor,
) -> torch.Tensor:
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: VarLenTensor,
    k: VarLenTensor,
    v: VarLenTensor,
) -> VarLenTensor:
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: VarLenTensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> VarLenTensor:
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    k: VarLenTensor,
    v: VarLenTensor,
) -> torch.Tensor:
    ...


def sparse_scaled_dot_product_attention(*args, **kwargs):
    num_all_args = len(args) + len(kwargs)

    if num_all_args == 1:
        names = ['qkv']
    elif num_all_args == 2:
        names = ['q', 'kv']
    elif num_all_args == 3:
        names = ['q', 'k', 'v']
    else:
        raise TypeError(
            f'Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3'
        )

    if len(args) > len(names):
        raise TypeError(
            f'Too many positional arguments, got {len(args)}, expected at most {len(names)}'
        )

    bound = {}
    for name, value in zip(names, args):
        bound[name] = value

    for key, value in kwargs.items():
        if key not in names:
            raise TypeError(f'Unexpected keyword argument: {key}')
        if key in bound:
            raise TypeError(f'Multiple values for argument: {key}')
        bound[key] = value

    missing = [name for name in names if name not in bound]
    if missing:
        raise TypeError(f'Missing argument(s): {", ".join(missing)}')

    if num_all_args == 1:
        qkv = bound['qkv']

        if not isinstance(qkv, VarLenTensor):
            raise TypeError(f'qkv must be a VarLenTensor, got {type(qkv)}')
        if len(qkv.shape) != 4 or qkv.shape[1] != 3:
            raise ValueError(
                f'Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]'
            )

        device = qkv.device
        s = qkv
        q_seqlen = [
            int(qkv.layout[i].stop - qkv.layout[i].start)
            for i in range(qkv.shape[0])
        ]
        kv_seqlen = list(q_seqlen)
        qkv = qkv.feats

    elif num_all_args == 2:
        q = bound['q']
        kv = bound['kv']

        valid = (
            isinstance(q, VarLenTensor)
            and isinstance(kv, (VarLenTensor, torch.Tensor))
        ) or (
            isinstance(q, torch.Tensor)
            and isinstance(kv, VarLenTensor)
        )
        if not valid:
            raise TypeError(
                f'Invalid types, got {type(q)} and {type(kv)}'
            )
        if q.shape[0] != kv.shape[0]:
            raise ValueError(
                f'Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}'
            )

        device = q.device

        if isinstance(q, VarLenTensor):
            if len(q.shape) != 3:
                raise ValueError(
                    f'Invalid shape for q, got {q.shape}, expected [N, *, H, C]'
                )
            s = q
            q_seqlen = [
                int(q.layout[i].stop - q.layout[i].start)
                for i in range(q.shape[0])
            ]
            q = q.feats
        else:
            if len(q.shape) != 4:
                raise ValueError(
                    f'Invalid shape for q, got {q.shape}, expected [N, L, H, C]'
                )
            s = None
            N, L, H, C = q.shape
            N = int(N)
            L = int(L)
            H = int(H)
            C = int(C)
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)

        if isinstance(kv, VarLenTensor):
            if len(kv.shape) != 4 or kv.shape[1] != 2:
                raise ValueError(
                    f'Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]'
                )
            kv_seqlen = [
                int(kv.layout[i].stop - kv.layout[i].start)
                for i in range(kv.shape[0])
            ]
            kv = kv.feats
        else:
            if len(kv.shape) != 5:
                raise ValueError(
                    f'Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]'
                )
            kvN, kvL, _, kvH, kvC = kv.shape
            kvN = int(kvN)
            kvL = int(kvL)
            kvH = int(kvH)
            kvC = int(kvC)
            kv_seqlen = [kvL] * kvN
            kv = kv.reshape(kvN * kvL, 2, kvH, kvC)

    else:
        q = bound['q']
        k = bound['k']
        v = bound['v']

        valid = (
            isinstance(q, VarLenTensor)
            and (
                (
                    isinstance(k, VarLenTensor)
                    and isinstance(v, VarLenTensor)
                )
                or (
                    isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                )
            )
        ) or (
            isinstance(q, torch.Tensor)
            and isinstance(k, VarLenTensor)
            and isinstance(v, VarLenTensor)
        )
        if not valid:
            raise TypeError(
                f'Invalid types, got {type(q)}, {type(k)}, and {type(v)}'
            )
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(
                f'Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}'
            )

        device = q.device

        if isinstance(q, VarLenTensor):
            if len(q.shape) != 3:
                raise ValueError(
                    f'Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]'
                )
            s = q
            q_seqlen = [
                int(q.layout[i].stop - q.layout[i].start)
                for i in range(q.shape[0])
            ]
            q = q.feats
        else:
            if len(q.shape) != 4:
                raise ValueError(
                    f'Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]'
                )
            s = None
            N, L, H, CI = q.shape
            N = int(N)
            L = int(L)
            H = int(H)
            CI = int(CI)
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)

        if isinstance(k, VarLenTensor):
            if not isinstance(v, VarLenTensor):
                raise TypeError(
                    f'Invalid types, got {type(k)} and {type(v)}'
                )
            if len(k.shape) != 3:
                raise ValueError(
                    f'Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]'
                )
            if len(v.shape) != 3:
                raise ValueError(
                    f'Invalid shape for v, got {v.shape}, expected [N, *, H, Co]'
                )
            if k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
                raise ValueError(
                    f'Key and value shape mismatch, got {k.shape} and {v.shape}'
                )

            kv_seqlen = [
                int(k.layout[i].stop - k.layout[i].start)
                for i in range(k.shape[0])
            ]
            v_seqlen = [
                int(v.layout[i].stop - v.layout[i].start)
                for i in range(v.shape[0])
            ]
            if kv_seqlen != v_seqlen:
                raise ValueError(
                    'Key and value sequence lengths must match'
                )

            k = k.feats
            v = v.feats
        else:
            if not isinstance(v, torch.Tensor):
                raise TypeError(
                    f'Invalid types, got {type(k)} and {type(v)}'
                )
            if len(k.shape) != 4:
                raise ValueError(
                    f'Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]'
                )
            if len(v.shape) != 4:
                raise ValueError(
                    f'Invalid shape for v, got {v.shape}, expected [N, L, H, Co]'
                )
            if (
                k.shape[0] != v.shape[0]
                or k.shape[1] != v.shape[1]
                or k.shape[2] != v.shape[2]
            ):
                raise ValueError(
                    f'Key and value shape mismatch, got {k.shape} and {v.shape}'
                )

            kvN, kvL, kvH, kvCI = k.shape
            kvN = int(kvN)
            kvL = int(kvL)
            kvH = int(kvH)
            kvCI = int(kvCI)
            kvCO = int(v.shape[-1])
            kv_seqlen = [kvL] * kvN

            k = k.reshape(kvN * kvL, kvH, kvCI)
            v = v.reshape(kvN * kvL, kvH, kvCO)

    if config.ATTN == 'xformers':
        import xformers.ops as xops

        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)[0]

    elif config.ATTN == 'flash_attn':
        import flash_attn

        cu_seqlens_q = torch.cat(
            [
                torch.tensor([0]),
                torch.cumsum(torch.tensor(q_seqlen), dim=0),
            ]
        ).int().to(device)

        if num_all_args in (2, 3):
            cu_seqlens_kv = torch.cat(
                [
                    torch.tensor([0]),
                    torch.cumsum(torch.tensor(kv_seqlen), dim=0),
                ]
            ).int().to(device)

        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens_q,
                max(q_seqlen),
            )
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                max(q_seqlen),
                max(kv_seqlen),
            )
        else:
            out = flash_attn.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max(q_seqlen),
                max(kv_seqlen),
            )

    elif config.ATTN == 'flash_attn_3':
        import flash_attn_interface as flash_attn_3

        cu_seqlens_q = torch.cat(
            [
                torch.tensor([0]),
                torch.cumsum(torch.tensor(q_seqlen), dim=0),
            ]
        ).int().to(device)

        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q.clone()
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max_q_seqlen
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = torch.cat(
                [
                    torch.tensor([0]),
                    torch.cumsum(torch.tensor(kv_seqlen), dim=0),
                ]
            ).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        else:
            cu_seqlens_kv = torch.cat(
                [
                    torch.tensor([0]),
                    torch.cumsum(torch.tensor(kv_seqlen), dim=0),
                ]
            ).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)

        out = flash_attn_3.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_q_seqlen,
            max_kv_seqlen,
        )

    elif config.ATTN == 'sdpa':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)

        out = _varlen_scaled_dot_product_attention(
            q,
            k,
            v,
            q_seqlen,
            kv_seqlen,
        )

    else:
        raise ValueError(f'Unknown attention module: {config.ATTN}')

    if isinstance(out, tuple):
        out = out[0]

    if s is not None:
        return s.replace(out)
    return out.reshape(N, L, H, -1)
