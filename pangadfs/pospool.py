# pangadfs/pangadfs/pospool.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Iterable
import pandas as pd

from pangadfs.base import PospoolBase


class PospoolDefault(PospoolBase):

    def pospool(self,
                *,
                pool: pd.DataFrame,
                posfilter: Dict[str, int],
                column_mapping: Dict[str, str],
                flex_positions: Iterable[str] = ('RB', 'WR', 'TE'),
                **kwargs
                ) -> Dict[str, pd.DataFrame]:
        """Creates initial position pool
        
        Args:
            pool (pd.DataFrame):
            posfilter (Dict[str, int]): filter out low scorers by position
            column_mapping (Dict[str, str]): column names for player, position, salary, projection
            flex_positions (Iterable[str]): e.g. (WR, RB, TE)
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            Dict[str, pd.DataFrame] where keys == posfilter.keys

        """
        d = {}
        poscol = column_mapping.get('position', 'pos')
        pointscol = column_mapping.get('points', 'proj')
        salcol = column_mapping.get('salary', 'salary')
        for position, thresh in posfilter.items():
            if position == 'FLEX':
                tmp = pool.loc[(pool[poscol].isin(flex_positions)) & (pool[pointscol] >= thresh), [pointscol, salcol]]      
            else:           
                tmp = pool.loc[(pool[poscol] == position) & (pool[pointscol] >= thresh), [pointscol, salcol]]
            prob_ = (tmp[pointscol] / tmp[salcol]) * 1000
            prob_ = prob_ / prob_.sum()
            d[position] = tmp.assign(prob=prob_)
        return d
