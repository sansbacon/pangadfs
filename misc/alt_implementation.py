import pandas as pd
import numpy as np

# constants
MIN_SALARIES = 49000
MAX_SALARIES = 50000
FLEX_POSITIONS = ('WR', 'TE', 'RB')
POSITION_RULES = (('QB', 1, 1), ('RB', 2, 3), ('WR', 3, 4), 
                  ('TE', 1, 2), ('DST', 1, 1), ('FLEX', 1, 1))
POSITION_THRESHOLDS = {'QB': 12, 'RB': 7, 'WR': 7, 'TE': 4, 'DST': 4}

# load data
df = pd.read_csv('C:/Users/erict/ewt/pangadfs/tests/players.csv')

# filter by projection thresholds (narrow search space)
fdf = pd.concat([df.loc[(df.pos == pos) & (df.proj > projmin), :] 
                 for pos, projmin in POSITION_THRESHOLDS.items()])
fdf['wval'] = np.sqrt(fdf['value'] * fdf['ceil'])
fdf['wval2'] = (fdf['value'] / 2) * (fdf['floor'] + fdf['proj'] + fdf['ceil'])
fdf['wval3'] = fdf['proj'] * fdf['salary']

# create initial position pool
pospool = {pos: fdf.loc[fdf.pos == pos, :] for pos in fdf.pos.unique()}

# add flex
pospool['FLEX'] = fdf.loc[fdf.pos.isin(FLEX_POSITIONS), :]
pospool['FLEX']['pos'] = 'FLEX'

# add weights
weightcol = 'wval3'
for pos, posdf in pospool.items():
    pospool[pos]['weights'] = posdf[weightcol] / posdf[weightcol].sum()

lu = []
existinglu = []

for i in range(25):
    while True:
        tmp = pd.concat([pospool[pos].sample(num, weights=pospool[pos]['weights']) 
                         for pos, num, _ in POSITION_RULES])
        if MIN_SALARIES < tmp['salary'].sum() < MAX_SALARIES:
            if tmp['proj'].sum() > MIN_FITNESS:
                ids = tuple(tmp.player_id.sort_values())
                if ids not in existinglu:
                    tmp['lid'] = i
                    lu.append(tmp)
                    existinglu.append(ids)
                    break
