# gadfs/app/settings.py

import os
from pathlib import Path

from dotenv import load_dotenv
from stevedore import driver


env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

plugin_names = {
  'crossover': os.getenv("PANGADFS_CROSSOVER_PLUGIN"),
  'populate': os.getenv("PANGADFS_POPULATE_PLUGIN"),
  'fitness': os.getenv("PANGADFS_FITNESS_PLUGIN"),
  'validate': os.getenv("PANGADFS_VALIDATE_PLUGIN"),
  'mutate': os.getenv("PANGADFS_MUTATE_PLUGIN"),
}

dmgrs = {}

for k, v in plugin_names.items():
    try:
        dmgrs[k] = driver.DriverManager(namespace=f'gadfs.{k}', name=v, invoke_on_load=False)
    except:
        pass

