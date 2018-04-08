from setuptools import setup

setup(
    name='tfhe',
    version='0.0.1',
    description='TFHE port in Python',
    url='http://github.com/nucypher/tfhe.py',
    author='Bogdan Opanchuk',
    author_email='bogdan@nucypher.com',
    license='MIT',
    packages=['tfhe'],
    install_requires=['numpy', 'reikna'],
    zip_safe=True)
