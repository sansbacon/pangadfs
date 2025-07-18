import pandas as pd
import numpy as np

# Load the sample data to see the structure
pool = pd.read_csv('pangadfs/app/appdata/pool.csv')
print("Pool columns:", pool.columns.tolist())
print("\nFirst few rows:")
print(pool.head())

print("\nPositions in data:")
print(pool['pos'].unique())

print("\nTE players:")
te_players = pool[pool['pos'] == 'TE']
print(te_players[['player', 'pos', 'salary', 'proj']].head())

# Simulate a lineup with TE
sample_lineup_indices = [0, 40, 45, 80, 85, 90, 95, 100, 105]  # Mix of positions
sample_lineup = pool.iloc[sample_lineup_indices]
print("\nSample lineup:")
print(sample_lineup[['player', 'pos', 'salary', 'proj']])

# Test the position mapping logic
players_by_pos = {}
for _, player in sample_lineup.iterrows():
    pos = player.get('pos', '')
    player_name = player.get('player', '')
    
    if pos not in players_by_pos:
        players_by_pos[pos] = []
    players_by_pos[pos].append(player_name)

print("\nPlayers by position:")
for pos, players in players_by_pos.items():
    print(f"{pos}: {players}")
