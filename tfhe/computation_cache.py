_computations = {}


def clean_arg(arg):
    if hasattr(arg, 'dtype'):
        return ('Type', arg.dtype, arg.shape, arg.strides, arg.offset)
    else:
        return arg


def clear_computation_cache():
    _computations.clear()


def get_computation(thr, cls, *args, **kwds):
    hashable_args = tuple(map(clean_arg, args))
    hashable_kwds = tuple((key, kwds[key]) for key in sorted(kwds))
    key = (id(thr), id(cls), hashable_args, hashable_kwds)
    if key in _computations:
        return _computations[key]
    else:
        comp = cls(*args, **kwds)
        compiled_comp = comp.compile(thr)
        _computations[key] = compiled_comp
        return compiled_comp
