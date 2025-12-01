from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="iactrace",
    version="0.1.0",
    author="Gerrit Roellinghoff",
    author_email="gerrit.roellinghoff@fau.de",
    description="JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/iactrace",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "trimesh>=3.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0",
        ],
        "gpu": [
            "jax[cuda13_pip]",
        ],
    },
)
