from setuptools import setup
from codecs import open
from os import path


from se3dif import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='cgdf',
      version=__version__,
      description='Constrained Grasp Diffusion Fields',
      author='Gaurav Singh',
      author_email='gaurav@brown.edu',
      packages=['cgdf'],
      install_requires=requires_list,
      )
