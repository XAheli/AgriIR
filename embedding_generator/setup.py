#!/usr/bin/env python3
"""
Setup script for Agriculture Embedding Generator
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agriculture-embedding-generator",
    version="1.0.0",
    author="Agriculture AI Team",
    author_email="contact@agriculture-ai.com",
    description="A comprehensive system for generating embeddings from agricultural text data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/agriculture-embedding-generator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "agriculture-embeddings=create_embeddings:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="agriculture, embeddings, nlp, machine-learning, semantic-search, faiss, transformers",
    project_urls={
        "Bug Reports": "https://github.com/your-org/agriculture-embedding-generator/issues",
        "Source": "https://github.com/your-org/agriculture-embedding-generator",
        "Documentation": "https://github.com/your-org/agriculture-embedding-generator/docs",
    },
)