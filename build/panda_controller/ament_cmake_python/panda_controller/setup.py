from setuptools import find_packages
from setuptools import setup

setup(
    name='panda_controller',
    version='0.0.0',
    packages=find_packages(
        include=('panda_controller', 'panda_controller.*')),
)
