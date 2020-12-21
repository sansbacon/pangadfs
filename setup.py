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
        description="extensible pandas-based library for NFL dfs genetic algorithm",
        author="Eric Truett",
        author_email="eric@erictruett.com",
        license="Apache 2.0",
        packages=find_packages(),
        entry_points={
          'pangadfs.crossover': ['crossover_default = pangadfs.default:DefaultCrossover'],
          'pangadfs.mutate': ['mutate_default = pangadfs.default:DefaultMutate'],
          'pangadfs.populate': ['populate_default = pangadfs.default:DefaultPopulate'],
          'pangadfs.fitness': ['fitness_default = pangadfs.default:DefaultFitness'],
          'pangadfs.validate': ['validate_default = pangadfs.default:DefaultValidate'],
          'pangadfs.pool': ['pool_default = pangadfs.default:DefaultPool'],
          'pangadfs.pospool': ['pospool_default = pangadfs.default:DefaultPospool'],
          'console_scripts': ['pangaopt=app.app:main']
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    run()
