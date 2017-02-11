# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:42:24 2016

@author: gsantiago
"""

import glob
import json
import datetime
from dateutil.relativedelta import relativedelta

files = glob.glob("*.json")

for f in files:
    with open(f) as jsonFile:
        data = json.load(jsonFile)
        initialDate = datetime.datetime.strptime(data['initialDate'], "%Y-%m-%d") + relativedelta(months=data['trainingSizeInMonths'])
        initialDate = datetime.datetime.strftime(initialDate, "%Y-%m-%d")
        
        print('{}_{}_{}_{}_{}'.format(data['trainingSizeInMonths'], data['windowSizeInMonths'], data['riskRate'], initialDate, data['finalDate']))
