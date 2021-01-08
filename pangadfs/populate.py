# pangadfs/pangadfs/populate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from pangadfs.base import PopulateBase
from pangadfs.misc import multidimensional_shifting


class PopulateDefault(PopulateBase):

    def populate(self,
                 *, 
                 pospool, 
                 posmap: Dict[str, int], 
                 population_size: int, 
                 probcol: str='prob') -> np.ndarray:
        """Creates individuals in population
        
        Args:
            pospool (Dict[str, DataFrame]): pool split into positions
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities

        Returns:
            ndarray of size (population_size, sum(posmap.values()))

        """
        pos_samples = {
            pos: multidimensional_shifting(pospool[pos].index, population_size, n, pospool[pos][probcol])
            for pos, n in posmap.items()
        }

        # concatenate positions into single row
        pop = np.concatenate([pos_samples[pos] for pos in posmap if pos != 'FLEX'], axis=1)

        # find non-duplicate FLEX and aggregate with other positions
        # https://stackoverflow.com/questions/65473095
        # https://stackoverflow.com/questions/54155844/
        dups = (pos_samples['FLEX'][..., None] == pop[:, None, :]).any(-1)
        return np.column_stack((pop, pos_samples['FLEX'][np.invert(dups).cumsum(axis=1).cumsum(axis=1) == 1]))