from setuptools import setup, find_packages


package_name = "prescyent"
__version__ = "0.3.0"


CORE_REQUIREMENTS = [
    "h5py>=3.10,<4",
    "matplotlib>=3.0,<3.8",
    "numpy>=1.17.3,<1.25.0",
    "pydantic>=2.0,<2.5",
    "pytorch_lightning>=2.0, <2.1",
    "scipy>=1.9,<1.12",
    "tensorboard>=2.0,<2.14",
    "torch>=2.0,<2.2",
    "tqdm>=4.66",
]


setup(
    name=package_name,
    version=__version__,
    description="Data-driven trajectory prediction library",
    long_description_content_type="text/markdown",
    url="https://github.com/hucebot/prescyent",
    author="Alexis Biver",
    author_email="alexis.biver@inria.fr",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=CORE_REQUIREMENTS,
)
