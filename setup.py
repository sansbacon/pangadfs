# -*- coding: utf-8 -*-
'''
setup.py

installation script

'''
from pathlib import Path
from setuptools import setup, find_packages


long_description = (Path(__file__).parent / 'README.md').read_text()


def run():
    setup(
        name='pangadfs',
        version='0.1',
        description='extensible pandas-based genetic algorithm for fantasy sports lineup optimization',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Eric Truett',
        author_email='sansbacon@gmail.com',
        license='MIT',
        packages=find_packages(),
        entry_points={
          'pangadfs.crossover': ['crossover_default = pangadfs.crossover:CrossoverDefault'],
          'pangadfs.select': ['select_default = pangadfs.select:SelectDefault'],
          'pangadfs.mutate': ['mutate_default = pangadfs.mutate:MutateDefault'],
          'pangadfs.populate': ['populate_default = pangadfs.populate:PopulateDefault'],
          'pangadfs.fitness': ['fitness_default = pangadfs.fitness:FitnessDefault'],
          'pangadfs.validate': ['validate_salary = pangadfs.validate:SalaryValidate',
                                'validate_duplicates = pangadfs.validate:DuplicatesValidate'],
          'pangadfs.pool': ['pool_default = pangadfs.pool:PoolDefault'],
          'pangadfs.pospool': ['pospool_default = pangadfs.pospool:PospoolDefault'],
          'console_scripts': ['basicapp=pangadfs.app.basicapp.app:run', 
                              'configapp=pangadfs.app.configapp.app:run']
        },
        zip_safe=False,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.8',
    )


if __name__ == '__main__':
    run()
