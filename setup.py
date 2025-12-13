from setuptools import setup, find_packages

setup(
    name="amp_bms",
    version="1.0",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
)
