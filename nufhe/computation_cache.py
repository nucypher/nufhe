# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from collections import defaultdict


_computations = defaultdict(lambda: dict())


def clean_arg(arg):
    if hasattr(arg, 'dtype'):
        return ('Type', arg.dtype, arg.shape, arg.strides, arg.offset)
    else:
        return arg


def clear_computation_cache(thr):
    """
    Clear the cache of computation objects compiled for the given ``reikna`` thread ``thr``.
    This will help ensure a correct realease of the thread's resources when the other references
    to it go out of scope
    (which is especially important for multi-threading applications using CUDA).

    .. note::

        :py:class:`~nufhe.Context` objects call this function automatically on destruction.
    """
    if id(thr) in _computations:
        del _computations[id(thr)]


def get_computation(thr, cls, *args, **kwds):
    hashable_args = tuple(map(clean_arg, args))
    hashable_kwds = tuple((key, kwds[key]) for key in sorted(kwds))
    key = (id(cls), hashable_args, hashable_kwds)
    if key in _computations[id(thr)]:
        return _computations[id(thr)][key]
    else:
        comp = cls(*args, **kwds)
        compiled_comp = comp.compile(thr)
        _computations[id(thr)][key] = compiled_comp
        return compiled_comp
