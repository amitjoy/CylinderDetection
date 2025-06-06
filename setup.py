from setuptools import setup, find_packages

setup(
    name="pycylinder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'open3d',
    ],
)
