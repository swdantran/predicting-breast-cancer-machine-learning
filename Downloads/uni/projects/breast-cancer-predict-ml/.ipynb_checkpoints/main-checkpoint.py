import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as msno 

import warnings
warnings.filterwarnings('ignore')

sns.set()
plt.style.use('ggplot')

df = pd.read_csv('breast-cancer.csv')
df.head()


