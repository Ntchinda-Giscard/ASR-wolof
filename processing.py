import pandas as pd
import json
import numpy as np
import re

train_file = 'dataset//train.json'

df = pd.read_json(train_file, lines=True)

print(df.head())