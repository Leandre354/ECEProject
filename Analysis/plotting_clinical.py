import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

#Loading the metrics and properties of baseline
db = pd.read_csv('Inference/Image200/Analysis/Baseline/Metrics_x_Clinical.csv', delimiter=';')

plt.figure('Clinical')
plt.subplot(1,2,1)
plt.boxplot(db['Likert'].iloc[::2])
plt.subplot(1,2,2)
plt.boxplot(db['Likert'].iloc[1::2])


plt.figure('Correlation')
plt.subplot(2,4,1).set_title('Nodal Likert v Dice')
plt.scatter(db['Likert'].iloc[::2], db['DICE'].iloc[::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['DICE'].iloc[::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,2).set_title('Nodal Likert v Hausdorff')
plt.scatter(db['Likert'].iloc[::2], db['HDRFDST'].iloc[::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['HDRFDST'].iloc[::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,3).set_title('Nodal Likert v Jaccard')
plt.scatter(db['Likert'].iloc[::2], db['JACRD'].iloc[::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['JACRD'].iloc[::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,4).set_title('Nodal Likert v Sensitivity')
plt.scatter(db['Likert'].iloc[::2], db['SNSVTY'].iloc[::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['SNSVTY'].iloc[::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,5).set_title('Primary Likert v Dice')
plt.scatter(db['Likert'].iloc[1::2], db['DICE'].iloc[1::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['DICE'].iloc[1::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,6).set_title('Primary Likert v Hausdorff')
plt.scatter(db['Likert'].iloc[1::2], db['HDRFDST'].iloc[1::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['HDRFDST'].iloc[1::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,7).set_title('Primary Likert v Jaccard')
plt.scatter(db['Likert'].iloc[1::2], db['JACRD'].iloc[1::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['JACRD'].iloc[1::2])
plt.text(1, 0.2,f'{corr:.3f}')
plt.subplot(2,4,8).set_title('Nodal Likert v Sensitivity')
plt.scatter(db['Likert'].iloc[1::2], db['SNSVTY'].iloc[1::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['SNSVTY'].iloc[1::2])
plt.text(1, 0.2,f'{corr:.3f}')

# Show the plot
plt.show()