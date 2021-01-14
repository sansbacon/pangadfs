# pangadfs/app/configapp/app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from dacite import from_dict
from stevedore.driver import DriverManager 
from stevedore.named import NamedExtensionManager

from gasettings import AppSettings
from optimizer import Optimizer


DATADIR = Path(__file__).parent.parent / 'appdata'


def main():
	"""Example application using pangadfs"""
	logging.basicConfig(level=logging.INFO)
	data = json.loads((Path(__file__).parent / 'config.json').read_text())
	ctx = from_dict(data_class=AppSettings, data=data)
	ctx.ga_settings.csvpth = DATADIR / ctx.ga_settings.csvpth

	dmgrs = {
      k: DriverManager(namespace=f'pangadfs.{k}', name=v, invoke_on_load=True)
      for k, v in ctx.plugin_settings.driver_managers.items()
	}

	emgrs = {
	  k: NamedExtensionManager(namespace=f'pangadfs.{k}', names=v, invoke_on_load=True, name_order=True)
      for k, v in ctx.plugin_settings.extension_managers.items()
	}

	# set up GeneticAlgorithm object
	opt = Optimizer(ctx=ctx, 
	                driver_managers=dmgrs, 
					extension_managers=emgrs)
	_ = opt.optimize(verbose=True)


if __name__ == '__main__':
	main()
