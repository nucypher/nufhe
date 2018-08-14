from setuptools import setup

setup(
    name='nufhe',
    version='0.0.1',
    description='A GPU implementation of the fully homomorphic encryption on torus',
    url='http://github.com/nucypher/nufhe',
    author='Bogdan Opanchuk',
    author_email='bogdan@nucypher.com',
    license='MIT',
    packages=['nufhe'],
    install_requires=['numpy', 'reikna>=0.7.1'],
    zip_safe=True)
