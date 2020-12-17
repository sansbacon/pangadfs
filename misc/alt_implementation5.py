# %%
import logging
import random
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# INITIAL VARIABLES
ISIZE = 5000
N_GENERATIONS = 20
TEAM_SIZE = 9
SALARY_CAP = 50000
BASE_POSITIONS = ('DST', 'QB', 'RB', 'TE', 'WR')
FLEX_POSITIONS = ('RB', 'TE', 'WR')
ALL_POSITIONS = ('DST', 'QB', 'RB', 'TE', 'WR', 'FLEX')
BASEDIR = Path.home() / 'workspace/nflprojections-data/2020/12/raw'


def bincount(a, valid_size=TEAM_SIZE):
    """Based on https://stackoverflow.com/questions/48473056/number-of-unique-elements-per-row-in-a-numpy-array"""
    n = a.max()+1
    a_off = a+(np.arange(a.shape[0])[:,None])*n
    M = a.shape[0]*n
    return np.where((np.bincount(a_off.ravel(), minlength=M).reshape(-1,n)!=0).sum(1) == valid_size, True, False)


def create_pospool(pool, FLEX_POSITIONS):
    """Creates position pool"""
    # create position pool
    # get standard positions
    # then special rules for flex
    wanted = ['proj', 'salary', 'ppd']
    posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
    pospool = {position: pool.loc[(pool.pos == position) & (pool.proj >= thresh), wanted]
                        for position, thresh in posfilter.items() if position != 'FLEX'}
    pospool['FLEX'] = pool.loc[(pool.pos.isin(FLEX_POSITIONS)) & (pool.proj >= posfilter['FLEX']), wanted]      
    return pospool
    
    
def create_pool(csvpth):
    """Creates player pool"""
    pool = pd.read_csv(csvpth)
    pool.columns = ['player', 'team', 'opp', 'pos', 'salary', 'proj', 'value', 'own', 'id']
    pool = pool.sort_values(['pos']).reset_index(drop=True)
    pool['value'] = pool['value'] + np.abs(pool['value'].min())
    pool['ppd'] = (pool.proj / pool.salary) * 1000
    pool['ppd'] = pool['ppd'] / pool['ppd'].sum()
    pool['wppd'] = ((pool['proj'] - pool['proj'].mean()) / pool['proj'].std()) * pool['ppd']
    pool['wppd'] = pool['wppd'] + np.abs(pool['wppd'].min())
    pool['wppd'] = pool['wppd'] / pool['wppd'].sum()
    return pool
    

def fitness(pop, ptsd):
    """Calculates population fitness"""
    return np.apply_along_axis(lambda x: sum([ptsd[i] for i in x]), axis=1, arr=pop)


def multidimensional_shifting(elements, num_samples, sample_size, probs):
    """Based on https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
    
    Args:
        elements (iterable): iterable to sample from, typically a dataframe index
        num_samples (int): the number of rows (e.g. initial population size)
        sample_size (int): the number of columns (e.g. team size)
        probs (iterable): is same size as elements

    Returns:
        ndarray of same shape as elements
        
    """
    replicated_probabilities = np.tile(probs, (num_samples, 1))
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    shifted_probabilities = random_shifts - replicated_probabilities
    return elements[np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]]


def populate(pospool, posmap, num_samples, probcol='ppd'):
    """Creates initial population"""
    # generate teams of IDs
    pos_samples = {
      pos: multidimensional_shifting(pospool[pos].index, num_samples, n, pospool[pos][probcol])
      for pos, n in posmap.items()
    }

    # concatenate positions into single row
    pop = np.concatenate([pos_samples[pos] for pos in BASE_POSITIONS], axis=1)

    # add flex and filter out duplicates
    flex = np.array([random.choice(np.setdiff1d(pos_samples['FLEX'][i], pop[i])) 
                               for i in range(num_samples)])
    return np.unique(np.column_stack((pop, flex)), axis=0)


def validate(pop, sald, salary_cap=SALARY_CAP):
    """Validates lineups against salary cap"""    
    # salary validation
    popsal = np.apply_along_axis(lambda x: sum([sald[i] for i in x]), axis=1, arr=pop)

    # duplicates validation
    pop = pop[popsal <= salary_cap]
    return pop[bincount(pop)]


# %%
def crossover(pop, pop_fitness, sald, pctl=50, mutation_rate=.05, mutation_pop=None):
    """Crosses over fittest pctl of population"""
    fittest = np.argwhere(pop_fitness > np.percentile(pop_fitness, pctl)).ravel()

    # ensure len(fittest) will split evenly
    if len(fittest) % 2 == 1:
        fittest = fittest[:-1]

    # split top 1/2 of population into parents
    fathers, mothers = train_test_split(pop[fittest], train_size=.5, test_size=.5)

    # choice is a 9-element array of True and False
    # swap rule ensure no duplicates
    choice = np.random.randint(2, size=fathers.size).reshape(fathers.shape).astype(bool)   
    newpop = np.where(choice, fathers, mothers)
    
    # now mutate
    if mutation_pop is not None:
        print(f'before mutation newpop is {newpop.shape}')
        mutation_pop = mutation_pop[:len(newpop)]
        mutate = np.random.binomial(n=1, p=mutation_rate, size=newpop.size).reshape(newpop.shape).astype(bool)
        newpop = np.where(choice, mutation_pop, newpop)
        print(f'after mutation newpop is {newpop.shape}')
    
    # validate new population
    newpop = validate(newpop, sald)
    return np.concatenate((pop[(-pop_fitness).argsort()[:ISIZE - len(newpop)]], newpop), axis=0)


# %%
# SETUP POOL AND POSITION POOL
pool = create_pool(BASEDIR / '2020_w12_etr_dk_main.csv')
pospool = create_pospool(pool, FLEX_POSITIONS)

# %%
# create dict of index and stat value
# this will allow easy lookup later on
ptsd = dict(zip(pool.index, pool.proj))
sald = dict(zip(pool.index, pool.salary))

# %%
# CREATE INITIAL POPULATION
# pos_samples are the index value
posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
pop = populate(pospool, posmap, ISIZE)

# %%
pop = validate(pop, sald)
pop_fitness = fitness(pop, ptsd)
maxidx = np.argmax(pop_fitness)
oldmax = pop_fitness[maxidx]
lu = pool.loc[pop[maxidx]]
print(lu)
print(round(oldmax, 2))

# CREATE NEW GENERATIONS
for _ in range(N_GENERATIONS):
    pop = crossover(pop, pop_fitness, sald)
    pop_fitness = fitness(pop, ptsd)
    maxidx = np.argmax(pop_fitness)
    thismax = pop_fitness[maxidx]
    if thismax > oldmax:
        lu = pool.loc[pop[maxidx]]
        print(lu)
        oldmax = thismax
        print(round(thismax, 2))
