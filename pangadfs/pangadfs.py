"""
pangadfs

pandas-based genetic algorithm for NFL dfs
currently based on DK, maybe more flexible in future

"""

# %%
import logging

import pandas as pd
import numpy as np


# %%
class PanGaDFS:

    DEFAULT_FLEX = ('RB', 'WR', 'TE')

    def __init__(self, player_pool, flex=DEFAULT_FLEX):
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

    @property
    def player_pool(self):
        if self._weighted_player_pool is None:
            self._weighted_player_pool = self.roulette_wheel()
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

    def _weighted_pos(self, pos):
        """Gets weighted position"""
        return self.player_pool.query(f'pos == "{pos}"')

    def create_lineup(self):
        """Creates lineup from df given rules"""
        # get position minimums + extra FLEXES
        return (
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
        )

    def roulette_wheel(self, projcol='proj'):
        """Creates weighted player pool by projection"""
        df = self._player_pool.copy()
        df['weight'] = 0
        for pos in df['pos'].unique():
            df.loc[df['pos'] == pos, 'weight'] = (
                df.loc[df['pos'] == pos, projcol] / df.loc[df['pos'] == pos, projcol]
                .sum()
            )
        return df

    def selection(self, df, n):
        """Initial selection of population"""
        pass

# %%
if __name__ == '__main__':
    pass