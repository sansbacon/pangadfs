# -*- coding: utf-8 -*-
"""
setup.py

installation script

"""

from setuptools import setup, find_packages

PACKAGE_NAME = "pangadfs"


def run():
    setup(
        name=PACKAGE_NAME,
        version="0.1",
        description="pandas-based library for NFL dfs genetic algorithm",
        author="Eric Truett",
        author_email="eric@erictruett.com",
        license="Apache 2.0",
        packages=find_packages(),
        entry_points={
          'gadfs.crossover': ['crossover_default = gadfs.default:DefaultCrossover'],
          'gadfs.mutate': ['mutate_default = gadfs.default:DefaultMutate'],
          'gadfs.populate': ['populate_default = gadfs.default:DefaultPopulate'],
          'gadfs.fitness': ['fitness_default = gadfs.default:DefaultFitness'],
          'gadfs.validate': ['validate_default = gadfs.default:DefaultValidate'],
          'console_scripts': ['gaopt=app.app:main']
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    run()
