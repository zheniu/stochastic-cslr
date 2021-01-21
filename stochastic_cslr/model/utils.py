def slice_at_dim(*args, dim=0):
    return [slice(None)] * max(dim, 0) + [slice(*args)]


def unpad_padded(x, xl, dim=0):
    return [xi[slice_at_dim(xli, dim=dim)] for xi, xli in zip(x, xl)]
