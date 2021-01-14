# pangadfs/pangadfs/pool.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict
from pathlib import Path

import pandas as pd

from pangadfs.base import PoolBase


class PoolDefault(PoolBase):

    def pool(self, 
             *, 
             csvpth: Path,
             thresh: int = 4,
             **kwargs
             ) -> pd.DataFrame:
        """Creates initial pool
        
        Args:
            csvpth (Path): path for csv file
            thresh (int): global filter on low-scoring players
            **kwargs: keyword arguments for plugins
            
        Returns:
            DataFrame with columns:
            player, team, pos, salary, proj

        """
        df = pd.read_csv(csvpth)
        assert set(['player', 'team', 'pos', 'salary', 'proj']).issubset(df.columns)
        df = df.loc[df.proj >= thresh, :]
        return df.sort_values(['pos']).reset_index(drop=True)

