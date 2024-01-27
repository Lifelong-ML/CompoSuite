from os import path
from setuptools import find_packages, setup

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="composuite",
    packages=[package for package in find_packages() if package.startswith("composuite")],
    install_requires=[
        "robosuite",
        "gym>=0.15.7",
        "h5py>=2.10.0",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3,<3.12",
    description="CompoSuite: A Compositional Reinforcement Learning Benchmark",
    author="Jorge Mendez, Marcel Hussing, Meghna Gummadi and Eric Eaton",
    url="https://github.com/Lifelong-ML/CompoSuite",
    author_email="",
    version="1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
