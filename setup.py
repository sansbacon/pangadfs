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
          'pangadfs.crossover': ['crossover_default = pangadfs.default:CrossoverDefault'],
          'pangadfs.mutate': ['mutate_default = pangadfs.default:MutateDefault'],
          'pangadfs.populate': ['populate_default = pangadfs.default:PopulateDefault'],
          'pangadfs.fitness': ['fitness_default = pangadfs.default:FitnessDefault'],
          'pangadfs.validate': ['validate_salary = pangadfs.default:SalaryValidate',
                                'validate_duplicates = pangadfs.default:DuplicatesValidate'],
          'pangadfs.pool': ['pool_default = pangadfs.default:PoolDefault'],
          'pangadfs.pospool': ['pospool_default = pangadfs.default:PospoolDefault'],
          'console_scripts': ['pangaopt=app.app:main']
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    run()
