from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="reproduce_tfidf",
    version="1.0",
    packages=[""],
    url="",
    license="",
    long_description=long_description,
    author="aimore",
    author_email="aimore.dutra@globality.com",
    description="Reproduce tfidf",
    install_requires=[
        "pandas==1.3.3",
        "scipy==1.7.1",
        "scikit-learn==1.0",
        "numpy==1.21.2",
        "ipykernel==6.7.0",
        "microcosm>=2.4.1",
    ],
)
