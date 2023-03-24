from setuptools import setup, find_packages

setup(
    name='nnscratch',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
    ],
    author='Karthikeya',
    author_email='yeliettikarthik0@gmail.com',
    description='A simple implementation of a neural network from scratch',
)