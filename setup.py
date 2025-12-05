#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NEK5000 Multi-Agent Reinforcement Learning Framework"

# Define dependencies based on imports found in the codebase
install_requires = [
    # Core scientific computing
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'pandas>=1.3.0',
    
    # Machine Learning and RL - Working version ranges
    'stable-baselines3>=1.7.0,<2.0.0',  # Compatible range
    'gym>=0.24.0,<0.25.0',  # Avoid problematic 0.21.0 build issues
    'pettingzoo>=1.19.0,<1.20.0',  # Compatible range
    'supersuit>=3.5.0,<3.6.0',  # Working range without BaseParallelWraper issue
    
    # Configuration management
    'omegaconf>=2.0.0',
    'pyyaml>=5.1.0',
    
    # MPI support
    'mpi4py>=3.1.0',
    
    # Visualization and logging
    'matplotlib>=3.4.0',
    'tensorboard>=2.8.0',
]

# Optional dependencies for development
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.910',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
    ],
}

# Entry points for command-line scripts
entry_points = {
    'console_scripts': [
        'nek-marl=nek_MARL.__main__:main',
    ],
}

setup(
    name='nek_MARL',
    version='0.0.1',
    author="Yuning Wang",
    author_email="yuninw@umich.edu",  # Update with your email
    description="NEK5000 Multi-Agent Reinforcement Learning Framework for Drag Reduction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Fantasy98/nek_drl",  # Update with your repository URL
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update with your license
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    include_package_data=True,
    zip_safe=False,
    keywords="reinforcement-learning, multi-agent, fluid-dynamics, nek5000, cfd",
    # This needs to be changed! 
    project_urls={
        "Bug Reports": "https://github.com/Fantasy98/nek_drl/issues",
        "Source": "https://github.com/yourusername/nek_drl",
        "Documentation": "https://nek-marl.readthedocs.io/",  # Update if you have docs
    },
)
