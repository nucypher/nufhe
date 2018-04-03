from reikna.core import Type


_computations = {}


def clean_arg(arg):
    if hasattr(arg, 'shape') and not isinstance(arg, Type):
        return Type(arg.dtype, arg.shape)
    else:
        return arg


def get_computation(thr, cls, *args, **kwds):
    args = tuple(map(clean_arg, args))
    key = (id(thr), args, kwds)
    if key in _computations:
        return _computations[key]
    else:
        comp = cls(*args, **kwds)
        compiled_comp = comp.compile(thr)
        _computations[key] = compiled_comp
        return compiled_comp
