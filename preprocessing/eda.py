import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/M01_Feb_2021_OP07_000_good.csv')

df[['x', 'y', 'z']].plot(subplots=True, sharex=True, figsize=(10, 8))

plt.xlabel('Index')
plt.tight_layout()
plt.show()