import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns

dane = pd.read_csv("shakespear.csv")

print("Median",np.median(dane["words"]))
print("Mean", np.mean(dane["words"]))
print("Mode", st.mode(dane["words"]))
print("Standard deviation", np.std(dane["words"]))
print("Variance", np.var(dane["words"]))


# Plotting data
sns.displot(dane, x="words")
sns.set(rc={'figure.figsize':(11,8)})
plt.title("Number of plays by word count")
plt.show()

