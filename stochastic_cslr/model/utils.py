def unpad_padded(x, xl, dim=0):
    dims = list(range(len(x.shape)))
    dims.insert(0, dims.pop(dim))
    x = x.permute(*dims)
    return [xi[:xli] for xi, xli in zip(x, xl)]
