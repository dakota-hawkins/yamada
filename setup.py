from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="yamada-mst",
    version="0.1.0",
    description="Package to find all minimum spanning trees in a network graph.",
    AUTHOR="Dakota Y. Hawkins",
    author_email="dyh0110@bu.edu",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/dakota-hawkins/yamada",
    install_requires=["networkx", "sortedcontainers", "numpy"],
    packages=["yamada"],
    license="MIT License",
)
