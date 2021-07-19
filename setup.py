from setuptools import setup, find_packages

setup(
    name = "jaxbase",
    packages=find_packages(),
    version = "0.1.0",
    author = "Alex Damian",
    author_email = "ad27@princeton.edu",
    description = "Jax utility package for common models and datasets",
    license = "MIT",
    keywords = "jax",
    # url = "http://packages.python.org/an_example_pypi_project",
    install_requires=[
        'jax',
        'flax',
        'tensorflow-datasets',
        'jax-resnet'
    ],
)