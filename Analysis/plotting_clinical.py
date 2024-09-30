import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from collections import Counter

#Loading the metrics and properties of baseline
db2 = pd.read_csv('Inference/Image200/Analysis/Baseline/Metrics_x_Clinical.csv', delimiter=';')
db = pd.read_csv('Inference/Image200/Analysis/Baseline/Metrics_x_Clinical_perturbed.csv', delimiter=';')

counts = Counter(db['Likert'].iloc[::2])
counts2 = Counter(db['Likert'].iloc[1::2])
counts3 = Counter(db2['Likert'].iloc[::2])
counts4 = Counter(db2['Likert'].iloc[1::2])

# Create lists of ratings and their corresponding counts
ratings = [1, 2, 3, 4, 5]
counts_list = [counts.get(rating, 0) for rating in ratings]
counts_list2 = [counts2.get(rating, 0) for rating in ratings]
counts_list3 = [counts3.get(rating, 0) for rating in ratings]
counts_list4 = [counts4.get(rating, 0) for rating in ratings]

# Plot the bar chart

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        absolute = int(np.round(pct * total / 100.0))
        return f'{absolute} ({pct:.1f}%)'
    return my_format

"""
plt.figure('Clinical')
plt.subplot(2,2,1).set_title('Nodal')
plt.pie(counts_list, labels=ratings, autopct=autopct_format(counts_list), textprops={'fontsize': 7})
plt.subplot(2,2,3).set_title('Primary')
plt.pie(counts_list2, labels=ratings, autopct=autopct_format(counts_list2), textprops={'fontsize': 7})
plt.subplot(2,2,2).set_title('Nodal perturbed')
plt.pie(counts_list3, labels=ratings, autopct=autopct_format(counts_list3), textprops={'fontsize': 7})
plt.subplot(2,2,4).set_title('Primary perturbed')
plt.pie(counts_list4, labels=ratings, autopct=autopct_format(counts_list4), textprops={'fontsize': 7})
"""

plt.figure('Correlation')
plt.subplot(2,3,1).set_title('Nodal Likert v Dice')
plt.scatter(db['Likert'].iloc[::2], db['DICE'].iloc[::2], marker='.')
m, b = np.polyfit(db['Likert'].iloc[::2], db['DICE'].iloc[::2], 1)
plt.plot(db['Likert'].iloc[::2], m * db['Likert'].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['DICE'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
plt.subplot(2,3,2).set_title('Nodal Likert v Hausdorff')
filtered_db = db[db['HDRFDST'] != 1000]
plt.scatter(filtered_db['Likert'].iloc[::2], filtered_db['HDRFDST'].iloc[::2], marker='.')
m, b = np.polyfit(filtered_db['Likert'].iloc[::2], filtered_db['HDRFDST'].iloc[::2], 1)
plt.plot(filtered_db['Likert'].iloc[::2], m * filtered_db['Likert'].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(filtered_db['Likert'].iloc[::2], filtered_db['HDRFDST'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
plt.subplot(2,3,3).set_title('Nodal Likert v Jaccard')
plt.scatter(db['Likert'].iloc[::2], db['JACRD'].iloc[::2], marker='.')
m, b = np.polyfit(db['Likert'].iloc[::2], db['JACRD'].iloc[::2], 1)
plt.plot(db['Likert'].iloc[::2], m * db['Likert'].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['JACRD'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
"""
plt.subplot(2,4,4).set_title('Nodal Likert v Sensitivity')
plt.scatter(db['Likert'].iloc[::2], db['SNSVTY'].iloc[::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[::2], db['SNSVTY'].iloc[::2])
plt.text(1, 0.2,f'{corr:.3f}')
"""
plt.subplot(2,3,4).set_title('Primary Likert v Dice')
plt.scatter(db['Likert'].iloc[1::2], db['DICE'].iloc[1::2], marker='.')
m, b = np.polyfit(db['Likert'].iloc[1::2], db['DICE'].iloc[1::2], 1)
plt.plot(db['Likert'].iloc[1::2], m * db['Likert'].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['DICE'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
plt.subplot(2,3,5).set_title('Primary Likert v Hausdorff')
filtered_db = db[db['HDRFDST'] != 1000]
plt.scatter(filtered_db['Likert'].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2], marker='.')
m, b = np.polyfit(filtered_db['Likert'].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2], 1)
plt.plot(filtered_db['Likert'].iloc[1::2], m * filtered_db['Likert'].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(filtered_db['Likert'].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
plt.subplot(2,3,6).set_title('Primary Likert v Jaccard')
plt.scatter(db['Likert'].iloc[1::2], db['JACRD'].iloc[1::2], marker='.')
m, b = np.polyfit(db['Likert'].iloc[1::2], db['JACRD'].iloc[1::2], 1)
plt.plot(db['Likert'].iloc[1::2], m * db['Likert'].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['JACRD'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')
"""
plt.subplot(2,4,8).set_title('Nodal Likert v Sensitivity')
plt.scatter(db['Likert'].iloc[1::2], db['SNSVTY'].iloc[1::2], marker='.')
corr, p = stats.pearsonr(db['Likert'].iloc[1::2], db['SNSVTY'].iloc[1::2])
plt.text(1, 0.2,f'{corr:.3f}')
"""

# Show the plot
plt.show()