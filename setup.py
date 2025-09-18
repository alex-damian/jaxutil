from setuptools import find_packages, setup

setup(
    name="jaxutil",
    packages=find_packages(),
    version="0.4.1",
    author="Alex Damian",
    author_email="ad27@princeton.edu",
    description="Jax utility package for common models and datasets",
    license="MIT",
    keywords="jax",
    install_requires=[
        "jax",
        "jaxopt",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
        "jax_tqdm",
        "treescope",
    ],
)
