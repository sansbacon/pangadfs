# pangadfs/pangadfs/populate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict

import numpy as np

from pangadfs.base import PopulateBase
from pangadfs.misc import multidimensional_shifting


class PopulateDefault(PopulateBase):

    def populate(self,
                 *, 
                 pospool, 
                 posmap: Dict[str, int], 
                 population_size: int, 
                 probcol: str='prob',
                 **kwargs) -> np.ndarray:
        """Creates individuals in population
        
        Args:
            pospool (Dict[str, DataFrame]): pool split into positions
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities
            **kwargs: keyword arguments

        Returns:
            ndarray of size (population_size, sum(posmap.values()))

        """
        pos_samples = {
            pos: multidimensional_shifting(pospool[pos].index, population_size, n, pospool[pos][probcol])
            for pos, n in posmap.items()
        }

        # Handle FLEX separately to maintain original behavior
        if 'FLEX' in posmap:
            # concatenate non-FLEX positions into single row
            non_flex_samples = [pos_samples[pos] for pos in posmap if pos != 'FLEX']
            if non_flex_samples:
                pop = np.concatenate(non_flex_samples, axis=1)
            else:
                pop = np.empty((population_size, 0), dtype=int)
            
            # For FLEX, take the correct number of positions as specified in posmap
            flex_count = posmap['FLEX']
            flex_sample = pos_samples['FLEX'][:, :flex_count]  # Take correct number of FLEX positions
            
            return np.column_stack((pop, flex_sample))
        else:
            # No FLEX position, concatenate all
            return np.concatenate([pos_samples[pos] for pos in posmap], axis=1)
