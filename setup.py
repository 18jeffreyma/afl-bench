from setuptools import setup

setup(
    name='afl_bench',
    version='0.1',
    author='TODO',
    author_email='TODO@TODO.COM',
    description='A benchmark for asynchronous federated learning',
    packages=['afl_bench'],
    install_requires=[
        'flwr',
        'torch',
        'torchvision',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
)
