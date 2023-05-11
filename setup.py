from setuptools import setup, find_packages
from prescyent import __version__

CORE_REQUIREMENTS = [
        "matplotlib",
        "numpy",
        "pydantic",
        "pytorch_lightning>=2.0.0",
        "tensorboard",
        "torch",
]

CORE_MODULES = ['prescyent.',
                'prescyent.dataset','prescyent.dataset.*',
                'prescyent.evaluator','prescyent.evaluator.*',
                'prescyent.predictor', 'prescyent.predictor.*',
                'prescyent.utils', 'prescyent.utils.*'
]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="prescyent",
    version=__version__,
    description="Data-driven trajectory prediction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hucebot/prescyent",
    author="Alexis Biver",
    author_email="alexis.biver@inria.fr",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=CORE_MODULES),
    include_package_data=True,
    install_requires=CORE_REQUIREMENTS
)