# pangadfs/tests/test_base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import pytest
from pangadfs.base import *


class Crossover(CrossoverBase):
    def t(self):
        return None


class Fitness(FitnessBase):
    def t(self):
        return None


class Mutate(MutateBase):
    def t(self):
        return None


class Pool(PoolBase):
    def t(self):
        return None


class Populate(PopulateBase):
    def t(self):
        return None


class Pospool(PospoolBase):
    def t(self):
        return None


class Validate(ValidateBase):
    def t(self):
        return None


def test_crossover_base():
    with pytest.raises(TypeError):
        Crossover()    


def test_fitness_base():
    with pytest.raises(TypeError):
        Fitness()


def test_mutate_base():
    with pytest.raises(TypeError):
        Mutate()


def test_populate_base():
    with pytest.raises(TypeError):
        Populate()


def test_pool_base():
    with pytest.raises(TypeError):
        Pool()


def test_pospool_base():
    with pytest.raises(TypeError):
        Pospool()


def test_validate_base():
    with pytest.raises(TypeError):
        Validate()
