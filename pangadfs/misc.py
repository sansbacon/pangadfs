# pangadfs/pangadfs/misc.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""Backward-compatible re-exports.

All functionality has been moved to focused modules:
- pangadfs.sampling: multidimensional_shifting, parents
- pangadfs.metrics: diversity, calculate_jaccard_diversity, exposure

This module re-exports everything for existing import paths.
"""

from pangadfs.sampling import multidimensional_shifting, parents  # noqa: F401
from pangadfs.metrics import diversity, calculate_jaccard_diversity, exposure  # noqa: F401
