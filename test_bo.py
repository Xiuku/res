import pandas as pd
# pyrefly: ignore [missing-import]
from edbo.bro import BO

parameters_space = {'temperature': [20, 40, 60, 80], 'concentration': [0.1, 0.2, 0.5, 1.0]}
historical_results = [{'temperature': 20, 'concentration': 0.1, 'yield': 15.0}, {'temperature': 40, 'concentration': 0.2, 'yield': 45.0}]
results_df = pd.DataFrame(historical_results)

import itertools
keys = list(parameters_space.keys())
values = list(parameters_space.values())
combinations = list(itertools.product(*values))
space_df = pd.DataFrame(combinations, columns=keys)

bo = BO(results=results_df, domain=space_df, target='yield')
print("BO initialized")
try:
    bo.run()
    print("BO run complete. Proposed:")
    print(bo.proposed_experiments)
except Exception as e:
    print("Error:", e)

