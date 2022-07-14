from setuptools import setup, find_packages

setup(
    name="jaxutil",
    packages=find_packages(),
    version="0.3.0",
    author="Alex Damian",
    author_email="ad27@princeton.edu",
    description="Jax utility package for common models and datasets",
    license="MIT",
    keywords="jax",
    install_requires=["jax", "flax", "jax-resnet", "GPUtil"],
)
