import itertools
import logging
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np


class PanGaDFS:

    DEFAULT_FLEX = ('RB', 'WR', 'TE')
    SALARY_CAP = 50000
    SALARY_FLOOR = 47500
    MIN_SCORE = 120

    def __init__(self, 
                 pool, 
                 flex=DEFAULT_FLEX, 
                 initial_size=500, 
                 mutation_rate=.05,
                 idcol='id',
                 ptscol='proj'):
        """Creates instance

        Args:
            player_pool (DataFrame): with minimum columns ...

        Returns:
            PanGaDFS

        """
        logging.getLogger('pangadfs').addHandler(logging.NullHandler())
        self.initial_size = initial_size
        self.idcol = idcol
        self.ptscol = ptscol
        pool = pool.assign(weight = self.weights(pool))
        self.pool = pool
        self.position_pool = self._create_position_pool()
       
    def _create_lineup_OLD(self, lineup_id=None, weightcol='weight'):
        """Creates lineup"""
        np.random.seed()
            
        # get position minimums + extra FLEXES
        posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1}
        dfs = [self.position_pool[pos].sample(n, weights=self.position_pool[pos][weightcol])
                   for pos, n in posmap.items()]
        dfs.append(self.position_pool['FLEX'].sample(8, weights=self.position_pool['FLEX'][weightcol]))
        lineup = (
          pd.concat(dfs)
          .drop_duplicates(subset=self.idcol)
          .iloc[0:9, :]
          .sort_values(self.idcol)
        )
        
        # now create 2 tuple - ids, points
        return tuple(lineup[self.idcol]), tuple(lineup[self.ptscol])
    
    def _create_lineup(self, lineup_id=None, weightcol='weight'):
        """Creates lineup"""
        np.random.seed()
            
        # get position minimums + extra FLEXES
        posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1}
        dfs = [self.position_pool[pos].sample(n)
                   for pos, n in posmap.items()]
        dfs.append(self.position_pool['FLEX'].sample(8))
        lineup = (
          pd.concat(dfs)
          .drop_duplicates(subset=self.idcol)
          .iloc[0:9, :]
          .sort_values(self.idcol)
        )
        
        # now create 2 tuple - ids, points
        return tuple(lineup[self.idcol]), tuple(lineup[self.ptscol])

    def _create_position_pool(self):
        """Creates position pool. Is dict, position is key, dataframe is value"""
        df = self.pool
        d = {}
        for position in ('QB', 'RB', 'WR', 'TE', 'DST'):
            d[position] = df.loc[df.pos == position, :]
        d['FLEX'] = df.loc[df.pos.isin(self.DEFAULT_FLEX), :]
        return d
        
    def _player_ids(self, df):
        """Creates player IDs"""
        return pd.Series(range(1, len(df) + 1))
    
    def mtdna(self, item1, item2):
        """Ensures limited overlap when breeding lineups"""
        return len(item1 & item2)
    
    def populate(self, use_multiprocessing=False):
        """Creates initial population"""
         = multidi
        #if use_multiprocessing:
        #with Pool(processes=cpu_count()) as p:
        #      lineups = p.map(self._create_lineup, range(self.initial_size))
        #    return {t[0]: t[1] for t in lineups}
        #return {t[0]: t[1] for t in [self._create_lineup() for _ in range(self.initial_size)]}     
                   
    def multidimensional_shifting(self, num_samples, sample_size, elements, probabilities):
        """Fast way to generate multiple sets of random indices
             For population generation, can create n samples with s elements
             Do it all at once, rather than continually sampling
             So, if initial population is 500, then num_samples is 500
               and sample_size is 1 for QB, 2 for RB, 3 for WR, etc.
        """       
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

    def weights(self, df, projcol='proj', salcol='salary'):
        """Weighted point-per-dollar
        
        Args:
            df (DataFrame): has columns for projections and salary
            projcol (str): the projections column, default 'proj'
            salcol (str): the salary column, default 'salary'

        """
        weights = (
          (df[projcol] - df[projcol].mean() / df[projcol].std()) * 
          (df[projcol] / df[salcol] * 1000)
        )
        
        return weights / weights.sum()
