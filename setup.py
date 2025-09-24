from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="two-sigma-financial-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A SOLID-principles based financial market prediction system using LightGBM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/two-sigma-financial-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "optimization": [
            "scikit-optimize>=0.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "two-sigma-predict=src.main:main",
        ],
    },
    keywords="machine-learning finance prediction lightgbm kaggle time-series",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/two-sigma-financial-prediction/issues",
        "Source": "https://github.com/yourusername/two-sigma-financial-prediction",
        "Documentation": "https://github.com/yourusername/two-sigma-financial-prediction#readme",
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
)
