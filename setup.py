import os
from setuptools import find_packages, setup

requirements = ['numpy==1.24.2', 'tqdm==4.66.1', 'pandas==2.1.3', 'ipykernel', 'ipywidgets', 
                'matplotlib==3.7.1', 'jupyter==1.0.0', 'torch==2.2.0', 'tensorboard==2.13.0', 'scipy', 'wandb',
                'ml_collections', 'pytest==7.4.3', 'tensordict']
setup(
    name='maenvs4vrp',
    packages=find_packages(where=['maenvs4vrp']),
    python_requires='>=3.10, <4',
    install_requires=requirements,
    version='0.1.0',
    description='Multi Agent Environments for Vehicle Routing Problems',
    author='mustelideos',
    license='',
)
