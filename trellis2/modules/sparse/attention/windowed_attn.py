from typing import *
import torch
import torch.nn.functional as F
import math
from .. import SparseTensor
from .. import config


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
    'sparse_windowed_scaled_dot_product_cross_attention',
]


def _to_int_list(seqlen: Union[List[int], Tuple[int, ...], torch.Tensor]) -> List[int]:
    """
    Convert a container of sequence lengths into a plain list of python ints.

    Args:
        seqlen (List[int] or Tuple[int, ...] or torch.Tensor): The sequence lengths.

    Returns:
        (List[int]): The sequence lengths as python ints.
    """
    if torch.is_tensor(seqlen):
        return [int(i) for i in seqlen.detach().cpu().reshape(-1).tolist()]
    return [int(i) for i in seqlen]


def _block_diagonal_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: Union[List[int], torch.Tensor],
    kv_seqlen: Optional[Union[List[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Exact block-diagonal (i.e. non-overlapping window) scaled dot product attention.

    The sequences are grouped by length, and every group is evaluated with a single batched
    `scaled_dot_product_attention` call. This is mathematically identical to
    `xops.memory_efficient_attention` with a `BlockDiagonalMask`, requires no third party
    package, and never pads, so no masked out position can contribute to a softmax.

    Args:
        q (torch.Tensor): A [T_Q, H, Ci] tensor with the queries of all windows concatenated.
        k (torch.Tensor): A [T_KV, H, Ci] tensor with the keys of all windows concatenated.
        v (torch.Tensor): A [T_KV, H, Co] tensor with the values of all windows concatenated.
        q_seqlen (List[int] or torch.Tensor): Length of every query window, in concatenation order.
        kv_seqlen (List[int] or torch.Tensor): Length of every key/value window. Defaults to q_seqlen.

    Returns:
        (torch.Tensor): A [T_Q, H, Co] tensor.
    """
    if kv_seqlen is None:
        kv_seqlen = q_seqlen
    q_seqlen = _to_int_list(q_seqlen)
    kv_seqlen = _to_int_list(kv_seqlen)
    assert len(q_seqlen) == len(kv_seqlen), \
        f'Number of query windows ({len(q_seqlen)}) and key/value windows ({len(kv_seqlen)}) must match'
    assert q.shape[0] == sum(q_seqlen), \
        f'Query tensor holds {q.shape[0]} tokens but the window lengths sum to {sum(q_seqlen)}'
    assert k.shape[0] == sum(kv_seqlen), \
        f'Key tensor holds {k.shape[0]} tokens but the window lengths sum to {sum(kv_seqlen)}'
    assert v.shape[0] == sum(kv_seqlen), \
        f'Value tensor holds {v.shape[0]} tokens but the window lengths sum to {sum(kv_seqlen)}'
    assert q.shape[1] == k.shape[1] == v.shape[1], \
        f'Head counts must match, got {q.shape[1]}, {k.shape[1]} and {v.shape[1]}'
    assert q.shape[2] == k.shape[2], \
        f'Query and key channel counts must match, got {q.shape[2]} and {k.shape[2]}'

    out_dtype = torch.promote_types(torch.promote_types(q.dtype, k.dtype), v.dtype)
    out = torch.zeros(q.shape[0], q.shape[1], v.shape[2], dtype=out_dtype, device=q.device)
    if len(q_seqlen) == 0:
        return out

    window_offsets: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        assert kv_len > 0 or q_len == 0, \
            'A non empty query window cannot attend to an empty key/value window'
        window_offsets.setdefault((q_len, kv_len), []).append((q_start, kv_start))
        q_start += q_len
        kv_start += kv_len

    for (q_len, kv_len), offsets in window_offsets.items():
        if q_len == 0:
            continue
        q_starts = torch.tensor([offset[0] for offset in offsets], dtype=torch.long, device=q.device)
        kv_starts = torch.tensor([offset[1] for offset in offsets], dtype=torch.long, device=k.device)
        q_index = q_starts.unsqueeze(1) + torch.arange(q_len, dtype=torch.long, device=q.device).unsqueeze(0)
        kv_index = kv_starts.unsqueeze(1) + torch.arange(kv_len, dtype=torch.long, device=k.device).unsqueeze(0)
        q_batched = q[q_index].transpose(1, 2)
        k_batched = k[kv_index].transpose(1, 2)
        v_batched = v[kv_index].transpose(1, 2)
        out_batched = F.scaled_dot_product_attention(q_batched, k_batched, v_batched)
        out_batched = out_batched.transpose(1, 2)
        out = out.index_put((q_index,), out_batched.to(out.dtype))

    return out


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (torch.Tensor): Sequence lengths.
        (dict): Attn func args.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else tuple(shift_window)
    window_size = (window_size,) * DIM if isinstance(window_size, int) else tuple(window_size)
    assert len(shift_window) == DIM, \
        f'Expected {DIM} shift values, got {len(shift_window)}'
    assert len(window_size) == DIM, \
        f'Expected {DIM} window sizes, got {len(window_size)}'
    assert all(ws > 0 for ws in window_size), \
        f'Window sizes must be positive, got {window_size}'

    shifted_coords = tensor.coords.clone().detach().long()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.long).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [max(1, math.ceil((int(mc) + 1) / int(ws))) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1], dtype=torch.long), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.long).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.long).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device, dtype=fwd_indices.dtype)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            global xops
            import xformers.ops as xops
        attn_func_args = {
            'attn_bias': xops.fmha.BlockDiagonalMask.from_seqlens(_to_int_list(seq_lens))
        }
    elif config.ATTN == 'flash_attn':
        attn_func_args = {
            'cu_seqlens': torch.cat([
                torch.zeros(1, device=seq_lens.device, dtype=seq_lens.dtype),
                torch.cumsum(seq_lens, dim=0)
            ], dim=0).int(),
            'max_seqlen': int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        }
    elif config.ATTN == 'sdpa':
        attn_func_args = {
            'seq_lens': seq_lens
        }
    else:
        raise ValueError(f'Unknown attention module: {config.ATTN}')

    return fwd_indices, bwd_indices, seq_lens, attn_func_args


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.

    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'windowed_attention_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, attn_func_args))
    else:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = serialization_spatial_cache

    qkv_feats = qkv.feats[fwd_indices]

    if config.DEBUG:
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_len = int(seq_lens[i])
            seq_coords = qkv_coords[start:start + seq_len]
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                    f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            start += seq_len

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            global xops
            import xformers.ops as xops
        q, k, v = qkv_feats.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            global flash_attn
            import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)
    elif config.ATTN == 'sdpa':
        q, k, v = qkv_feats.unbind(dim=1)
        out = _block_diagonal_scaled_dot_product_attention(q, k, v, attn_func_args['seq_lens'])
    else:
        raise ValueError(f'Unknown attention module: {config.ATTN}')

    out = out[bwd_indices]

    if config.DEBUG:
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


def sparse_windowed_scaled_dot_product_cross_attention(
    q: SparseTensor,
    kv: SparseTensor,
    q_window_size: int,
    kv_window_size: int,
    q_shift_window: Tuple[int, int, int] = (0, 0, 0),
    kv_shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    """
    Apply windowed scaled dot product cross attention to two sparse tensors.

    Args:
        q (SparseTensor): [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor): [N, *, 2, H, C] sparse tensor containing Ks and Vs.
        q_window_size (int): The window size to use for Qs.
        kv_window_size (int): The window size to use for Ks and Vs.
        q_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Qs.
        kv_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Ks and Vs.

    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
    assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"

    q_serialization_spatial_cache_name = f'windowed_attention_{q_window_size}_{q_shift_window}'
    q_serialization_spatial_cache = q.get_spatial_cache(q_serialization_spatial_cache_name)
    if q_serialization_spatial_cache is None:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = calc_window_partition(q, q_window_size, q_shift_window)
        q.register_spatial_cache(q_serialization_spatial_cache_name, (q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args))
    else:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = q_serialization_spatial_cache
    kv_serialization_spatial_cache_name = f'windowed_attention_{kv_window_size}_{kv_shift_window}'
    kv_serialization_spatial_cache = kv.get_spatial_cache(kv_serialization_spatial_cache_name)
    if kv_serialization_spatial_cache is None:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = calc_window_partition(kv, kv_window_size, kv_shift_window)
        kv.register_spatial_cache(kv_serialization_spatial_cache_name, (kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args))
    else:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = kv_serialization_spatial_cache

    assert len(q_seq_lens) == len(kv_seq_lens), "Number of sequences in q and kv must match"

    q_feats = q.feats[q_fwd_indices]
    kv_feats = kv.feats[kv_fwd_indices]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            global xops
            import xformers.ops as xops
        k_feats, v_feats = kv_feats.unbind(dim=1)
        q_batched = q_feats.unsqueeze(0)
        k_batched = k_feats.unsqueeze(0)
        v_batched = v_feats.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(_to_int_list(q_seq_lens), _to_int_list(kv_seq_lens))
        out = xops.memory_efficient_attention(q_batched, k_batched, v_batched, attn_bias=mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            global flash_attn
            import flash_attn
        out = flash_attn.flash_attn_varlen_kvpacked_func(
            q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )
    elif config.ATTN == 'sdpa':
        k_feats, v_feats = kv_feats.unbind(dim=1)
        out = _block_diagonal_scaled_dot_product_attention(
            q_feats, k_feats, v_feats,
            q_attn_func_args['seq_lens'], kv_attn_func_args['seq_lens'],
        )
    else:
        raise ValueError(f'Unknown attention module: {config.ATTN}')

    out = out[q_bwd_indices]

    return q.replace(out)
