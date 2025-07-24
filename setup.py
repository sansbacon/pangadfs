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
        version='0.2',
        description='extensible pandas-based genetic algorithm for fantasy sports lineup optimization',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Eric Truett',
        author_email='sansbacon@gmail.com',
        license='MIT',
        packages=find_packages(),
        entry_points={
          'console_scripts': ['scaffold-plugin = pangadfs.scaffold_plugin:main'],
          'pangadfs.crossover': ['crossover_default = pangadfs.crossover:CrossoverDefault',
                                 'crossover_multilineup_sets = pangadfs.crossover_sets:CrossoverMultilineupSets'],
          'pangadfs.fitness': ['fitness_default = pangadfs.fitness:FitnessDefault',
                               'fitness_multilineup_sets = pangadfs.fitness_sets:FitnessMultilineupSets'],
          'pangadfs.mutate': ['mutate_default = pangadfs.mutate:MutateDefault',
                             'mutate_multilineup_sets = pangadfs.mutate_sets:MutateMultilineupSets'],
          'pangadfs.optimize': ['optimize_default = pangadfs.optimize:OptimizeDefault',
                                'optimize_multilineup = pangadfs.optimize:OptimizeMultilineup',
                                'optimize_multilineup_sets = pangadfs.optimize:OptimizeMultilineupSets',
                                'optimize_pool_based_sets = pangadfs.optimize_pool_based:OptimizePoolBasedSets',
                                'optimize_multi_objective = pangadfs.optimize_multi_objective:OptimizeMultiObjective'],
          'pangadfs.penalty': ['distance_penalty = pangadfs.penalty.DistancePenalty',
                               'diversity_penalty = pangadfs.penalty.DiversityPenalty',
                               'ownership_penalty = pangadfs.penalty.OwnershipPenalty',
                               'high_ownership_penalty = pangadfs.penalty.HighOwnershipPenalty',
                               ],
          'pangadfs.pool': ['pool_default = pangadfs.pool:PoolDefault'],
          'pangadfs.populate': ['populate_default = pangadfs.populate:PopulateDefault',
                               'populate_multilineup_sets = pangadfs.populate_sets:PopulateMultilineupSets'],
          'pangadfs.pospool': ['pospool_default = pangadfs.pospool:PospoolDefault'],
          'pangadfs.select': ['select_default = pangadfs.select:SelectDefault'],
          'pangadfs.validate': ['validate_salary = pangadfs.validate:SalaryValidate',
                                'validate_duplicates = pangadfs.validate:DuplicatesValidate',
                                'validate_positions = pangadfs.validate_positions:PositionValidate']
        },
        zip_safe=False,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.10',
    )


if __name__ == '__main__':
    run()
