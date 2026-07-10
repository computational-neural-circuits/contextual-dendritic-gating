from setuptools import find_packages, setup

setup(
    name="contextual-dendritic-gating",
    version="1.0.0",
    description="Brian2 simulations of contextual dendritic gating and neural assemblies",
    author="Sebastian Onasch, Christoph Miehl, M. Maurycy Miekus",
    packages=find_packages(),
    package_data={"src": ["model_specs/*.txt"]},
    python_requires=">=3.10",
    install_requires=[
        "brian2==2.9.0",
        "Cython>=3.2,<3.3",
        "h5py>=3.15,<3.16",
        "matplotlib>=3.10,<3.11",
        "networkx>=3.4,<3.5",
        "numpy>=2.0,<2.3",
        "python-louvain==0.16",
        "scikit-learn>=1.7,<1.8",
        "scipy>=1.15,<1.16",
    ],
    extras_require={
        "figures": [
            "matplotlib-venn==1.1.2",
            "torch==2.2.2",
            "torchvision==0.17.2",
        ],
        "dev": [
            "black==26.5.1",
            "ruff==0.15.21",
        ],
    },
)
