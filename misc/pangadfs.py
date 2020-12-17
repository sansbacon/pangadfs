"""
pangadfs

pandas-based genetic algorithm for NFL dfs
currently based on DK, maybe more flexible in future

"""

# %%
import itertools
import logging
from multiprocessing import Pool, cpu_count
import random
import uuid

import pandas as pd
import numpy as np


# %%
class PanGaDFS:

    DEFAULT_FLEX = ('RB', 'WR', 'TE')
    SALARY_CAP = 50000
    SALARY_FLOOR = 47500
    MIN_SCORE = 120

    def __init__(self, 
                 player_pool, 
                 flex=DEFAULT_FLEX, 
                 initial_size=500, 
                 mutation_rate=.15,
                 projcol='proj',
                 roulette_method='proj'):
        """Creates instance

        Args:
            player_pool (DataFrame): with minimum columns ...

        Returns:
            PanGaDFS

        """
        logging.getLogger(__file__).addHandler(logging.NullHandler())
        self.flex_eligible = flex
        self._player_pool = player_pool
        self._weighted_player_pool = None
        self._weighted_qbs = None
        self._weighted_rbs = None
        self._weighted_wrs = None
        self._weighted_tes = None
        self._weighted_dsts = None
        self._weighted_flex = None
        self.lineups = None
        self.initial_size = initial_size
        self.mutation_rate = mutation_rate
        self.fitness_func = None
        self.projcol = projcol
        self.roulette_method = roulette_method
        
    @property
    def player_pool(self):
        if self._weighted_player_pool is None:
            self._weighted_player_pool = self.roulette_wheel(self.roulette_method)
        return self._weighted_player_pool

    @property
    def dsts(self):
        if self._weighted_dsts is None:
            self._weighted_dsts = self._weighted_pos('DST')
        return self._weighted_dsts

    @property
    def flex(self):
        if self._weighted_flex is None:
            self._weighted_flex = self.player_pool.query("pos in @self.flex_eligible")
        return self._weighted_flex

    @property
    def qbs(self):
        if self._weighted_qbs is None:
            self._weighted_qbs = self._weighted_pos('QB')
        return self._weighted_qbs

    @property
    def rbs(self):
        if self._weighted_rbs is None:
            self._weighted_rbs = self._weighted_pos('RB')
        return self._weighted_rbs

    @property
    def tes(self):
        if self._weighted_tes is None:
            self._weighted_tes = self._weighted_pos('TE')
        return self._weighted_tes

    @property
    def wrs(self):
        if self._weighted_wrs is None:
            self._weighted_wrs = self._weighted_pos('WR')
        return self._weighted_wrs

    def _breed_two(self, lu, mother_or_father=None, max_attempts=10):
        """Breeds two lineups
           Tries max_attempts, if fail then breed with new lineup
        """
        attempts = 0
        while attempts < max_attempts:
            # randomly assign father/mother for each lineup slot
            # this should be a 9-item array of zeros and ones            
            if not mother_or_father:
                mother_or_father = np.random.binomial(1, .5, size=len(lu[0]))

            # picks mother or father for each position
            # have to use index slice to get dataframe instead of series
            new_lineup = pd.concat([
                lu[parent].iloc[idx:idx+1, :]
                for idx, parent in enumerate(mother_or_father)
            ])

            # end loop if valid lineup, assign new uuid
            # no need to assign id or calc fitness if lineup invalid
            if self.is_valid_lineup(new_lineup):
                new_lineup['lid'] = uuid.uuid4()
                new_lineup['fitness'] = self.fitness(new_lineup)
                return new_lineup
            attempts += 1
        return self._breed_two((random.choice(lu), self.create_lineup()))

    def _weighted_pos(self, pos):
        """Gets weighted position"""
        return self.player_pool.query(f'pos == "{pos}"')

    def breed(self):
        """Breeds lineups together
        """
        new_lineups = []

        # give weight in breeding to higher-fitness lineups
        lineup_fitness = (
            self.lineups
            .groupby('lid', sort=False)
            .first()
            .loc[:, ['fitness']]
        )

        weights = lineup_fitness['fitness'] / lineup_fitness['fitness'].sum()

        # make new lineups
        lu = self.lineups
        for _ in range(len(lineup_fitness)):
            # weighted selection of lineup ids by fitness
            lid1, lid2 = lineup_fitness.sample(2, weights=weights).index

            # select the two lineups and breed
            lu1 = lu.loc[lu.lid == lid1, :]
            lu2 = lu.loc[lu.lid == lid2, :]
            new_lineup = self._breed_two((lu1, lu2))

            # apply mutation according to mutation rate
            # add lineup
            new_lineups.append(self.mutate(new_lineup))

        return pd.concat(new_lineups)

    def create_lineup(self, lineup_id=None):
        """Creates lineup from df given rules"""
        np.random.seed()
        while True:
            # supply unique lineup id if none specified
            if not lineup_id:
                lineup_id = uuid.uuid4()

            # get position minimums + extra FLEXES
            lineup = (
              pd.concat([
                self.qbs.sample(1, weights=self.qbs['weight']),
                self.rbs.sample(2, weights=self.rbs['weight']),
                self.wrs.sample(3, weights=self.wrs['weight']),
                self.tes.sample(1, weights=self.tes['weight']),
                self.dsts.sample(1, weights=self.dsts['weight']),
                self.flex.sample(8, weights=self.flex['weight'])
              ])
              .drop_duplicates(subset='player_id')
              .iloc[0:9, :]
              .assign(lid=lineup_id)
            )

            # return if valid lineup
            if self.is_valid_lineup(lineup):
                return lineup

    def fitness(self, lineups):
        """Assesses fitness of individual"""
        if not self.fitness_func:
            return lineups.groupby('lid')[self.projcol].transform('sum')       
        else:
            return self.fitness_func(lineups)

    def initial_population(self):
        """Creates initial population of lineups"""
        with Pool(processes=cpu_count()) as p:
            lineups = p.map(self.create_lineup, range(1, self.initial_size + 1))
        self.lineups = pd.concat(lineups)
        self.lineups['fitness'] = self.fitness(self.lineups)
        return self.lineups

    def is_valid_lineup(self, lineup):
        """Tests if valid lineup"""
        sal = lineup.salary.sum()
        proj = lineup.proj.sum()
        return ((self.SALARY_CAP >= sal and
                 sal >= self.SALARY_FLOOR) and
                 proj >= self.MIN_SCORE and
                 len(lineup.player_id.unique()) == 9)

    def mutate(self, lineup):
        """Mutates lineup"""
        mf = np.random.binomial(1, .05, size=9)
        if 1 in mf:
            lu = (lineup, self.create_lineup())
            return self._breed_two(lu, mother_or_father=mf)
        return lineup

    def roulette_wheel(self, projcol='proj'):
        """Creates weighted player pool by projection
           Can do on entire player pool as pandas will normalize weights
        """
        df = self._player_pool.copy()
        if projcol == 'composite':
            # create composite weight from floor/mean/ceil
            # convert to value and scale into weight
            composite = np.sqrt((df['floor']**2 + df['ceil']**2 + df['proj'] **2) / 3)
            value = composite / df['salary'] * 1000
            df['weight'] = value / value.sum()
        else:
            df['weight'] = df[projcol] / df[projcol].sum()
        return df


# %%
if __name__ == '__main__':
    pass