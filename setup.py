import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="entmax-jax",
    version="0.1.0",
    author="Patrick Fernandes",
    author_email="pattuga@gmail.com",
    description="The entmax mapping in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-spin/entmax",
    packages=setuptools.find_packages(),
    install_requires=["jax>=0.1.75"],
    extras_require={"dev": ["jaxlib", "pytest", "flake8", "black"]},
    python_requires=">=3.6",
)
