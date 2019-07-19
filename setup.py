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

from setuptools import setup

setup(
    name='nufhe',
    version='0.0.3',
    description='A GPU implementation of fully homomorphic encryption on torus',
    url='http://github.com/nucypher/nufhe',
    author='Bogdan Opanchuk',
    author_email='bogdan@nucypher.com',
    license='GPLv3',
    packages=[
        'nufhe',
        'nufhe/transform',
        ],
    package_data={
        'nufhe': ['*.mako'],
        'nufhe/transform': ['*.mako'],
        },
    install_requires=['numpy', 'reikna>=0.7.4'],
    python_requires='>=3.5',
    extras_require={
        'dev': ['pytest', 'pytest-cov', 'sphinx', 'sphinx_autodoc_typehints'],
        'pyopencl': ['pyopencl>=2018.1.1'],
        'pycuda': ['pycuda>=2018.1.1'],
        },
    zip_safe=True)
