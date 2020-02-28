from setuptools import find_packages, setup

setup(
    name="mltools",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "sklearn",
        "pandas",
        "beautifulsoup4",
        "requests",
        "mecab-python3",
    ],
)
