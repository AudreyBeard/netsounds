from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='netsounds',
      version='0.0.2',
      description='An exploration of neural networks through sound',
      long_description=long_description,
      author='Audrey Beard',
      author_email='audrey.s.beard@gmail.com',
      packages=find_packages(),
      url='https://github.com/AudreyBeard/netsounds',
      changelog={'0.0.0': 'First-pass implementation with Nunpy',
                 '0.0.1': 'Support for PyTorch'
                 }
      )
