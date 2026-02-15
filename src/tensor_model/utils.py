def unbroadcast(grad, target_shape):
    # On réduit le gradient pour qu'il corresponde à la forme d'origine
    res = grad
    # (10, 5) vs (5,)
    while res.ndim > len(target_shape):
        res = res.sum(axis=0)
    # (10, 5) vs (1, 5)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            res = res.sum(axis=i, keepdims=True)
    return res