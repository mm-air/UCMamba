import torch
import warnings

# WITH_TRITON = True
WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)        
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

# Added by Shirley: spiral
def spiral_indices(H, W, device):
    """
    Efficiently generate spiral indices for an H x W image.
    Returns row_idx, col_idx (each shape [H*W]), uses pure Python but once per shape.
    """
    # Preallocate arrays, fill via numpy if needed for large images (optional)
    indices = []
    top, bottom, left, right = 0, H - 1, 0, W - 1
    while top <= bottom and left <= right:
        indices.extend([(top, col) for col in range(left, right + 1)])
        top += 1
        indices.extend([(row, right) for row in range(top, bottom + 1)])
        right -= 1
        if top <= bottom:
            indices.extend([(bottom, col) for col in range(right, left - 1, -1)])
            bottom -= 1
        if left <= right:
            indices.extend([(row, left) for row in range(bottom, top - 1, -1)])
            left += 1

    idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    return idx_tensor[:, 0], idx_tensor[:, 1]

def spiral_cw_gather(tensor_img, in_channel_first=True):
    """
    Vectorized spiral gather for (B, C, H, W) or (B, H, W, C), any H/W.
    Output: (B, C, H*W) or (B, H*W, C)
    """
    if in_channel_first:
        B, C, H, W = tensor_img.shape
    else:
        B, H, W, C = tensor_img.shape
    row_idx, col_idx = spiral_indices(H, W, tensor_img.device)
    # Use broadcasting for batch and channel dims
    if in_channel_first:
        # Output: (B, C, H*W)
        output = tensor_img[:, :, row_idx, col_idx]
    else:
        # Output: (B, H*W, C)
        output = tensor_img[:, row_idx, col_idx, :]
    return output

def spiral_cw_scatter(tensor_flat, original_shape, out_channel_first=True):
    """
    Vectorized spiral scatter for any image shape, batch, and channel size.
    Input: (B, C, H*W) or (B, H*W, C)
    Output: (B, C, H, W) or (B, H, W, C)
    """
    if out_channel_first:
        B, C, H, W = original_shape
    else:
        B, H, W, C = original_shape
    row_idx, col_idx = spiral_indices(H, W, tensor_flat.device)
    if out_channel_first:
        result_img = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
        result_img[:, :, row_idx, col_idx] = tensor_flat
    else:
        result_img = torch.zeros(B, H, W, C, device=tensor_flat.device, dtype=tensor_flat.dtype)
        result_img[:, row_idx, col_idx, :] = tensor_flat
    return result_img

def cross_scan_fwd_cw(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0: #6D: # H + V + S
            y = x.new_empty((B, 6, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)  # horizontal
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # vertical
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # horizontal & vertical flip

            # Add the scan for spiral direction
            y[:, 4, :, :] = spiral_cw_gather(x)  # spiral
            y[:, 5, :, :] = torch.flip(y[:, 4, :, :], dims=[-1])  # spiral flip

        elif scans == 1: #4D: H + V
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)  # horizontal
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # vertical
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # horizontal & vertical flip

        elif scans == 2: #4D: H + S
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)  # horizontal
            y[:, 1, :, :] = spiral_cw_gather(x)  # spiral
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # horizontal & spiral flip

        elif scans == 3: #4D: S + V
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = spiral_cw_gather(x)  # spiral
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # vertical
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # vertical & spiral flip

        elif scans == 4: #2D: S
            y = x.new_empty((B, 2, C, H * W))
            y[:, 0, :, :] = spiral_cw_gather(x)  # spiral
            y[:, 1, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])

        elif scans == 5: #2D: H
            y = x.new_empty((B, 2, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)  # horizontal
            y[:, 1, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])

        elif scans == 6: #2D: V
            y = x.new_empty((B, 2, C, H * W))
            y[:, 0, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # vertical
            y[:, 1, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])

        elif scans == 7:  # 1D: S
            #y = spiral_cw_gather(x).repeat(1, 2, 1, 1)  # spiral
            y = x.new_empty((B, 1, C, H * W))
            y[:, 0, :, :] = spiral_cw_gather(x)

        elif scans == 8:  #1D: H
            y = x.new_empty((B, 1, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)  # horizontal

        elif scans == 9:  #1D: H
            y = x.new_empty((B, 1, C, H * W))
            y[:, 0, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # vertical

    else:  # in_channel_first = False (B, H, W, C)

        B, H, W, C = x.shape
        L = H * W
        scan_h = x.flatten(1, 2)  #horizontal
        scan_v = x.transpose(1, 2).flatten(1, 2) #vertical

        if scans == 0:  # H + V + S
            y = x.new_empty((B, L, 6, C))
            y[:, :, 0, :] = scan_h
            y[:, :, 1, :] = scan_v
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
            scan_s = spiral_cw_gather(x, in_channel_first=False)
            y[:, :, 4, :] = scan_s
            y[:, :, 5, :] = torch.flip(scan_s, dims=[1])

        elif scans == 1:  # H + V
            y = x.new_empty((B, L, 4, C))
            y[:, :, 0, :] = scan_h
            y[:, :, 1, :] = scan_v
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])

        elif scans == 2:  # H + S
            y = x.new_empty((B, L, 4, C))
            y[:, :, 0, :] = scan_h
            scan_s = spiral_cw_gather(x, in_channel_first=False)
            y[:, :, 1, :] = scan_s
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])

        elif scans == 3:  # S + V
            y = x.new_empty((B, L, 4, C))
            scan_s = spiral_cw_gather(x, in_channel_first=False)
            y[:, :, 0, :] = scan_s
            y[:, :, 1, :] = scan_v
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])

        elif scans == 4:  # S
            y = x.new_empty((B, L, 2, C))
            scan_s = spiral_cw_gather(x, in_channel_first=False)
            y[:, :, 0, :] = scan_s
            y[:, :, 1, :] = torch.flip(scan_s, dims=[1])

        elif scans == 5:  # H
            y = x.new_empty((B, L, 2, C))
            y[:, :, 0, :] = scan_h
            y[:, :, 1, :] = torch.flip(scan_h, dims=[1])

        elif scans == 6:  # V
            y = x.new_empty((B, L, 2, C))
            y[:, :, 0, :] = scan_v
            y[:, :, 1, :] = torch.flip(scan_v, dims=[1])

        elif scans == 7:  # S1d
            #scan_s = spiral_cw_gather(x, in_channel_first=False)
            #y = scan_s.repeat(1, 1, 2, 1)
            y = x.new_empty((B, L, 1, C))
            y[:, :, 0, :] = spiral_cw_gather(x, in_channel_first=False)

        elif scans == 8:  # H1D
            y = x.new_empty((B, L, 1, C))
            y[:, :, 0, :] = scan_h

        elif scans == 9:  # V1D
            y = x.new_empty((B, L, 1, C))
            y[:, :, 0, :] = scan_v

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

def cross_merge_fwd_cw(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        L = H * W
        if scans == 0:  #6D
            y1 = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y1 = y1[:, 0] + y1[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1) #Convert the vertical parts to horizontal -> add them together -> convert them back to the original matrix form.
            y1 = y1.view(B, -1, H, W)
            y2 = y[:, 4] + y[:, 5].flip(dims=[-1]).view(B, D, -1)
            y2 = spiral_cw_scatter(y2, (B, D, H, W))
            y = y1 + y2

        elif scans == 1:  # 4D: H+V (Standard Cross Scan)
            y_h = y[:, 0] + y[:, 2].flip(dims=[-1])
            y_h = y_h.view(B, D, H, W)

            y_v = y[:, 1] + y[:, 3].flip(dims=[-1])
            y_v = y_v.view(B, D, W, H).transpose(dim0=2, dim1=3).contiguous()
            y = y_h + y_v

        elif scans == 2:  # 4D: H+S
            y_h = y[:, 0] + y[:, 2].flip(dims=[-1])
            y_h = y_h.view(B, D, H, W)
            y_s = y[:, 1] + y[:, 3].flip(dims=[-1])
            y_s = spiral_cw_scatter(y_s, (B, D, H, W))
            y = y_h + y_s

        elif scans == 3:  # 4D: V+S
            y_s = y[:, 0] + y[:, 2].flip(dims=[-1])
            #y_s = y_s.view(B, D, W, H).transpose(dim0=2, dim1=3).contiguous()
            y_s = spiral_cw_scatter(y_s, (B, D, H, W))
            y_v = y[:, 1] + y[:, 3].flip(dims=[-1])
            #y_v = spiral_cw_scatter(y_v, (B, D, H, W))
            y_v = y_v.view(B, D, W, H).transpose(dim0=2, dim1=3).contiguous()
            y = y_v + y_s

        elif scans == 4:  # 2D: S
            y_s = y[:, 0] + y[:, 1].flip(dims=[-1])
            y = spiral_cw_scatter(y_s, (B, D, H, W))

        elif scans == 5:  # 2D: H
            y_h = y[:, 0] + y[:, 1].flip(dims=[-1])
            y = y_h.view(B, D, H, W)

        elif scans == 6:  # 2D: V
            y_v = y[:, 0] + y[:, 1].flip(dims=[-1])
            y = y_v.view(B, D, W, H).transpose(dim0=2, dim1=3).contiguous()

        elif scans == 7:  # 1D: S
            #y_s = y[:, 0] + y[:, 1]   # y_s = y.sum(2)
            #y = spiral_cw_scatter(y_s, (B, D, H, W))
            y = spiral_cw_scatter(y[:, 0], (B, D, H, W))

        elif scans == 8:  # 1D: H
            y_h = y[:, 0]
            y = y_h.view(B, D, H, W)

        elif scans == 9:  # 1D: V
            y_v = y[:, 0]
            y = y_v.view(B, D, W, H).transpose(dim0=2, dim1=3).contiguous()

    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y1 = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y1 = y1[:, :, 0] + y1[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)
            y1 = y1.view(B, H, W, -1)
            y2 = y[:, :, 4] + y[:, :, 5].flip(dims=[1]).view(B, -1, D)
            y2 = spiral_cw_scatter(y2, (B, H, W, D), out_channel_first=False)
            y = y1 + y2

        elif scans == 1:  # H+V
            y_h = y[:, :, 0] + y[:, :, 2].flip(dims=[1])
            y_h = y_h.view(B, H, W, D)
            y_v = y[:, :, 1] + y[:, :, 3].flip(dims=[1])
            y_v = y_v.view(B, W, H, D).transpose(dim0=1, dim1=2).contiguous()
            y = y_h + y_v

        elif scans == 2:  # H+S
            y_h = y[:, :, 0] + y[:, :, 2].flip(dims=[1])
            y_h = y_h.view(B, H, W, D)
            y_s = y[:, :, 1] + y[:, :, 3].flip(dims=[1])
            y_s = spiral_cw_scatter(y_s, (B, H, W, D), out_channel_first=False)
            y = y_h + y_s

        elif scans == 3:  # S+V
            y_s = y[:, :, 0] + y[:, :, 2].flip(dims=[1])
            #y_s = y_s.view(B, W, H, D).transpose(dim0=1, dim1=2).contiguous()
            y_s = spiral_cw_scatter(y_s, (B, H, W, D), out_channel_first=False)
            y_v = y[:, :, 1] + y[:, :, 3].flip(dims=[1])
            #y_v = spiral_cw_scatter(y_v, (B, H, W, D), out_channel_first=False)
            y_v = y_v.view(B, W, H, D).transpose(dim0=1, dim1=2).contiguous()
            y = y_v + y_s

        elif scans == 4:  # S
            y_s = y[:, :, 0] + y[:, :, 1].flip(dims=[1])
            y = spiral_cw_scatter(y_s, (B, H, W, D), out_channel_first=False)

        elif scans == 5:  # H
            y_h = y[:, :, 0] + y[:, :, 1].flip(dims=[1])
            y = y_h.view(B, H, W, D)

        elif scans == 6:  # V
            y_v = y[:, :, 0] + y[:, :, 1].flip(dims=[1])
            y = y_v.view(B, W, H, D).transpose(dim0=1, dim1=2).contiguous()

        elif scans == 7:  # S1D
            #y_s = y[:, :, 0] + y[:, :, 1]  # y_s = y.sum(3)
            #y = spiral_cw_scatter(y_s, (B, H, W, D), out_channel_first=False)
            y = spiral_cw_scatter(y[:,:, 0], (B, H, W, D), out_channel_first=False)

        elif scans == 8:  #H1D
            y_h = y[:, :, 0]
            y = y_h.view(B, H, W, D)

        elif scans == 9:  # V
            y_v = y[:, :, 0]
            y = y_v.view(B, W, H, D).transpose(dim0=1, dim1=2).contiguous()

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y

class CrossScanF_cw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        #if one_by_one:
        #    B, K, C, H, W = x.shape
        #    if not in_channel_first:
        #        B, H, W, K, C = x.shape
        #else:
        B, C, H, W = x.shape
        if not in_channel_first:
            B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        #_fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        _fn = cross_scan_fwd_cw
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        #_fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        _fn = cross_merge_fwd_cw
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        #if one_by_one:
        #    y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        #else:
        y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF_cw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)

        #_fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        _fn = cross_merge_fwd_cw
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        #if not one_by_one:
        if in_channel_first:
            x = x.view(B, C, H, W)
        else:
            x = x.view(B, H, W, C)
        #else:
        #    if in_channel_first:
        #        x = x.view(B, 4, C, H, W)
        #    else:
        #        x = x.view(B, H, W, 4, C)

        #_fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        _fn = cross_scan_fwd_cw
        x = _fn(x, in_channel_first, out_channel_first, scans)
        #x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
        #x = x.view(B, 6, C, H, W) if out_channel_first else x.view(B, H, W, 6, C)
        x = x.view(B, -1, C, H, W) if out_channel_first else x.view(B, H, W, -1, C)

        return x, None, None, None, None

class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
    
        return x, None, None, None, None


# triton implements ========================================

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor, # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h # trans
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = neg_w * DH + neg_h # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = HWRoute2      

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
    
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1  
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC        
    
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, C, H, W, NH, NW
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
        
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    #CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    CSF = CrossScanF_cw
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    #CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    CMF = CrossMergeF_cw
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# checks =================================================================

class CHECK:
    def check_csm_triton():
        B, C, H, W = 256, 192, 56, 57
        dtype=torch.float16
        dtype=torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def unidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
            return x
        
        def unidi_merge(ys):
            B, K, C, H, W = ys.shape
            return ys.view(B, 4, -1, H * W).sum(1)

        def bidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x
        
        def bidi_merge(ys):
            B, K, D, H, W = ys.shape
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            # channel_first -> channel_first
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            
            # flex: BLC->BCL; BCL->BLC; BLC->BLC;
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
            
            # previous
            # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
            # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
            # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
            # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        if False:
            x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
            out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
            out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 1)
            out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            print(out1.view(4, -1))
            print(out2.view(-1))
            print(out3.view(4, -1))
            print(out4.view(4, -1))
            print(out5.view(-1, 4).t())
            print(out6.view(-1))
            print(out7.view(-1, 4).t())
            print(out8.view(-1, 4).t())

        B, C, H, W = 27, 253, 57, 58
        x = torch.randn((B, C, H, W)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, H, W)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            x2 = x.permute(0, 2, 3, 1).contiguous()
            o2 = o1.permute(0, 3, 4, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

        breakpoint()


if __name__ == "__main__":
    CHECK.check_csm_scan3()
    CHECK.check_csm_triton()




