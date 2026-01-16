#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="agri-bot-searcher",
    version="1.0.0",
    author="Agriculture Bot Team",
    author_email="contact@agribot.ai",
    description="Multi-agent agriculture chatbot with web search and citations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agriculture-bot/agri-bot-searcher",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Agriculture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agri-bot=src.agriculture_chatbot:main",
            "agri-bot-web=src.web_api:run_server",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="agriculture chatbot AI machine-learning farming agricultural-technology",
    project_urls={
        "Bug Reports": "https://github.com/agriculture-bot/agri-bot-searcher/issues",
        "Source": "https://github.com/agriculture-bot/agri-bot-searcher",
        "Documentation": "https://github.com/agriculture-bot/agri-bot-searcher/tree/main/docs",
    },
)
