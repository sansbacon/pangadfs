# pangadfs/select.py

import numpy as np
from pangadfs.base import SelectBase


class SelectDefault(SelectBase):

    def _fittest(self,
                 *,
                 population: np.ndarray, 
                 population_fitness: np.ndarray,
                 n: int,
                 **kwargs) -> np.ndarray:
        """n-fittest selection of individuals in population. 
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select

        Returns:
            np.ndarray: selected population

        """
        # argpartition produces an unsorted rearrangement
        # use negative parameter and slice to get top n values
        return population[np.argpartition(population_fitness, -n, axis=0)][-n:]

    def _rank(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               **kwargs) -> np.ndarray:
        """Rank selection of individuals in population. 
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select

        Returns:
            np.ndarray: selected population

        """
        elements = len(population)
        temp = population_fitness.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1, elements + 1)
        weights = ranks / ranks.sum()
        return population[np.random.choice(elements, size=n, replace=False, p=weights)]

    def _roulette_wheel(self, 
                        *, 
                        population: np.ndarray, 
                        population_fitness: np.ndarray,
                        n: int,
                        **kwargs) -> np.ndarray:
        """Select individuals in population using stochastic universal sampling, which samples without replacement.
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select

        Returns:
            np.ndarray: selected population

        """
        weights = population_fitness / population_fitness.sum()
        return population[np.random.choice(len(population_fitness), size=n, replace=True, p=weights)]       

    def _scaled(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               **kwargs) -> np.ndarray:
        """Select individuals in population using scaled fitness.
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select

        Returns:
            np.ndarray: selected population

        """
        scaled = (population_fitness - population_fitness.min()) / (population_fitness.max() - population_fitness.min())
        weights = scaled / scaled.sum()
        return population[np.random.choice(len(population), size=n, replace=False, p=weights)]

    def _sus(self, 
             *, 
             population: np.ndarray, 
             population_fitness: np.ndarray,
             n: int,
             **kwargs) -> np.ndarray:
        """Select individuals in population using stochastic universal sampling, which samples without replacement.
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select

        Returns:
            np.ndarray: selected population

        """
        weights = population_fitness / population_fitness.sum()
        return population[np.random.choice(len(population), size=n, replace=False, p=weights)]
     
    def _tournament(self, 
                    *, 
                    population: np.ndarray, 
                    population_fitness: np.ndarray,
                    tournament_size: int = 2,
                    **kwargs) -> np.ndarray:
        """Tournament selection of individuals in population. 
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            tournament_size (int): number of individuals to compete against each other

        Returns:
            np.ndarray: selected population

        """
        max_idx = len(population_fitness) // tournament_size * tournament_size
        tournament_fitness = population_fitness[:max_idx].reshape(-1, tournament_size)
        winners = np.argmax(tournament_fitness, axis=1) + np.arange(max_idx, step=tournament_size)
        return population[winners]

    def select(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               method: str = 'roulette',
               **kwargs) -> np.ndarray:
        """Select individuals in population using specified method.
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            method (str): 'roulette', 'su', 'scaled'; default 'roulette'

        Returns:
            np.ndarray: selected population

        """
        dispatch = {
            'roulette': self._roulette_wheel,
            'sus': self._sus,
            'rank': self._rank,
            'tournament': self._tournament,
            'scaled': self._scaled,
            'fittest': self._fittest
        }        

        params = {
            'population': population,
            'population_fitness': population_fitness,
            'n': n
        }

        return dispatch.get(method, self._roulette_wheel)(**{**params, **kwargs})
