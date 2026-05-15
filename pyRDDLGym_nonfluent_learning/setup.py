# This file is part of pyRDDLGym.

# pyRDDLGym is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# pyRDDLGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with pyRDDLGym. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
      name='pyRDDLGym-nonfluent-learning',
      version='0.1',
      author="Michael Gimelfarb, Scott Sanner",
      author_email="mike.gimelfarb@mail.utoronto.ca, ssanner@mie.utoronto.ca",
      description="pyRDDLGym-nonfluent-learning: non-fluent learning module for pyRDDLGym-jax.",
      license="MIT License",
      url="https://github.com/pyrddlgym-project/pyRDDLGym-nonfluent-learning",
      packages=find_packages(),
      install_requires=[
          'pyRDDLGym>=2.7',
          'pyRDDLGym-jax>=3.1',
          'rddlrepository>=2.2',
          'blackjax>=1.5',
      ],
      python_requires=">=3.12",
      package_data={'': ['*.cfg', '*.ico']},
      include_package_data=True,
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
