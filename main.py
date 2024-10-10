# This program predicts the unit sales for thousands of items sold at different Favorita stores.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


train=pd.read_csv('train.csv',
                  index_col='family',
                  )
sales=train.loc[:,'sales']

ax=sales.plot(figsize=(11,5),style='.',color='0.5',title='Daily Sales',legend=False)

plt.show()