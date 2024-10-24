"""
File: clinical_plot.py

Description:
This script implements the plottings for the clinical evaluation
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from collections import Counter

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        absolute = int(np.round(pct * total / 100.0))
        return f'{absolute} ({pct:.1f}%)'
    return my_format

#Loading the metrics and properties of baseline
db = pd.read_csv('Inference/Image200/Analysis/Baseline/Metrics_x_Clinical.csv', delimiter=';')
db2 = pd.read_csv('Inference/Image200/Analysis/Baseline/Metrics_x_Clinical_perturbed.csv', delimiter=';')

observer = 'Likert 1'

dice_bins = [0, 0.20, 0.40, 0.60, 0.80, 1]
dice_labels = ['0-0.20', '0.20-0.40', '0.40-0.60', '0.60-0.80', '0.80-1']

# Categorize the DICE values into the defined bins for both databases
db['DICE_category'] = pd.cut(db['DICE'], bins=dice_bins, labels=dice_labels, include_lowest=True)
db2['DICE_category'] = pd.cut(db2['DICE'], bins=dice_bins, labels=dice_labels, include_lowest=True)

# Count occurrences in each DICE category for even and odd indices
counts_dice1 = Counter(db['DICE_category'].iloc[::2])
counts_dice2 = Counter(db['DICE_category'].iloc[1::2])
counts_dice3 = Counter(db2['DICE_category'].iloc[::2])
counts_dice4 = Counter(db2['DICE_category'].iloc[1::2])

# Create lists of counts for the pie chart
counts_list_dice1 = [counts_dice1.get(category, 0) for category in dice_labels]
counts_list_dice2 = [counts_dice2.get(category, 0) for category in dice_labels]
counts_list_dice3 = [counts_dice3.get(category, 0) for category in dice_labels]
counts_list_dice4 = [counts_dice4.get(category, 0) for category in dice_labels]

# Plot the pie charts
plt.figure('DICE Distribution')

# Nodal (db, even indices)
plt.subplot(2, 2, 1).set_title('Nodal DICE')
plt.pie(counts_list_dice1, labels=dice_labels, autopct=autopct_format(counts_list_dice1), textprops={'fontsize': 7})

# Primary (db, odd indices)
plt.subplot(2, 2, 3).set_title('Primary DICE')
plt.pie(counts_list_dice2, labels=dice_labels, autopct=autopct_format(counts_list_dice2), textprops={'fontsize': 7})

# Nodal perturbed (db2, even indices)
plt.subplot(2, 2, 2).set_title('Nodal perturbed DICE')
plt.pie(counts_list_dice3, labels=dice_labels, autopct=autopct_format(counts_list_dice3), textprops={'fontsize': 7})

# Primary perturbed (db2, odd indices)
plt.subplot(2, 2, 4).set_title('Primary perturbed DICE')
plt.pie(counts_list_dice4, labels=dice_labels, autopct=autopct_format(counts_list_dice4), textprops={'fontsize': 7})

counts = Counter(db[observer].iloc[::2])
counts2 = Counter(db[observer].iloc[1::2])
counts3 = Counter(db2[observer].iloc[::2])
counts4 = Counter(db2[observer].iloc[1::2])

# Create lists of ratings and their corresponding counts
ratings = [1, 2, 3, 4, 5]
counts_list = [counts.get(rating, 0) for rating in ratings]
counts_list2 = [counts2.get(rating, 0) for rating in ratings]
counts_list3 = [counts3.get(rating, 0) for rating in ratings]
counts_list4 = [counts4.get(rating, 0) for rating in ratings]

# Plot the bar chart
plt.figure('Clinical')
plt.subplot(2,2,1).set_title('Nodal')
plt.pie(counts_list, labels=ratings, autopct=autopct_format(counts_list), textprops={'fontsize': 7})
plt.subplot(2,2,3).set_title('Primary')
plt.pie(counts_list2, labels=ratings, autopct=autopct_format(counts_list2), textprops={'fontsize': 7})
plt.subplot(2,2,2).set_title('Nodal perturbed')
plt.pie(counts_list3, labels=ratings, autopct=autopct_format(counts_list3), textprops={'fontsize': 7})
plt.subplot(2,2,4).set_title('Primary perturbed')
plt.pie(counts_list4, labels=ratings, autopct=autopct_format(counts_list4), textprops={'fontsize': 7})


plt.figure('Correlation')

plt.subplot(2,3,1).set_title('Nodal Likert v Dice')
plt.scatter(db[observer].iloc[::2], db['DICE'].iloc[::2], marker='.')
m, b = np.polyfit(db[observer].iloc[::2], db['DICE'].iloc[::2], 1)
plt.plot(db[observer].iloc[::2], m * db[observer].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db[observer].iloc[::2], db['DICE'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2,3,2).set_title('Nodal Likert v Hausdorff')
filtered_db = db[db['HDRFDST'] != 1000]
plt.scatter(filtered_db[observer].iloc[::2], filtered_db['HDRFDST'].iloc[::2], marker='.')
m, b = np.polyfit(filtered_db[observer].iloc[::2], filtered_db['HDRFDST'].iloc[::2], 1)
plt.plot(filtered_db[observer].iloc[::2], m * filtered_db[observer].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(filtered_db[observer].iloc[::2], filtered_db['HDRFDST'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2,3,3).set_title('Nodal Likert v Jaccard')
plt.scatter(db[observer].iloc[::2], db['JACRD'].iloc[::2], marker='.')
m, b = np.polyfit(db[observer].iloc[::2], db['JACRD'].iloc[::2], 1)
plt.plot(db[observer].iloc[::2], m * db[observer].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db[observer].iloc[::2], db['JACRD'].iloc[::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2,3,4).set_title('Primary Likert v Dice')
plt.scatter(db[observer].iloc[1::2], db['DICE'].iloc[1::2], marker='.')
m, b = np.polyfit(db[observer].iloc[1::2], db['DICE'].iloc[1::2], 1)
plt.plot(db[observer].iloc[1::2], m * db[observer].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db[observer].iloc[1::2], db['DICE'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2,3,5).set_title('Primary Likert v Hausdorff')
filtered_db = db[db['HDRFDST'] != 1000]
plt.scatter(filtered_db[observer].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2], marker='.')
m, b = np.polyfit(filtered_db[observer].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2], 1)
plt.plot(filtered_db[observer].iloc[1::2], m * filtered_db[observer].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(filtered_db[observer].iloc[1::2], filtered_db['HDRFDST'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2,3,6).set_title('Primary Likert v Jaccard')
plt.scatter(db[observer].iloc[1::2], db['JACRD'].iloc[1::2], marker='.')
m, b = np.polyfit(db[observer].iloc[1::2], db['JACRD'].iloc[1::2], 1)
plt.plot(db[observer].iloc[1::2], m * db[observer].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db[observer].iloc[1::2], db['JACRD'].iloc[1::2])
xlim = plt.xlim()
ylim = plt.ylim()
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')


plt.figure('Physician correlation')
plt.subplot(2,2,1).set_title('Nodal Observer 1 v Observer 2')
plt.scatter(db['Likert 1'].iloc[::2], db['Likert 2'].iloc[::2], marker='.')
m, b = np.polyfit(db['Likert 1'].iloc[::2], db['Likert 2'].iloc[::2], 1)
plt.plot(db['Likert 1'].iloc[::2], m * db['Likert 1'].iloc[::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert 1'].iloc[::2], db['Likert 2'].iloc[::2])
xlim = plt.xlim(0.8, 5.2)
ylim = plt.ylim(0.8, 5.2)
plt.xlabel('Observer 1 Ratings')
plt.ylabel('Observer 2 Ratings')
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2, 2, 2).set_title('Heatmap of Rating Agreement')
heatmap_data = pd.crosstab(db['Likert 2'].iloc[::2], db['Likert 1'].iloc[::2])  # Create a crosstab for the heatmap
sns.heatmap(heatmap_data, annot=True, cmap='Reds', cbar=True, fmt='d')
plt.xlabel('Observer 1 Ratings')
plt.ylabel('Observer 2 Ratings')
plt.gca().invert_yaxis()  # Inverts the Y-axis

plt.subplot(2,2,3).set_title('Primary Observer 1 v Observer 2')
plt.scatter(db['Likert 1'].iloc[1::2], db['Likert 2'].iloc[1::2], marker='.')
m, b = np.polyfit(db['Likert 1'].iloc[1::2], db['Likert 2'].iloc[1::2], 1)
plt.plot(db['Likert 1'].iloc[1::2], m * db['Likert 1'].iloc[1::2] + b, color='red', label=f'Trendline: y={m:.2f}x+{b:.2f}')
corr, p = stats.pearsonr(db['Likert 1'].iloc[1::2], db['Likert 2'].iloc[1::2])
xlim = plt.xlim(0.8, 5.2)
ylim = plt.ylim(0.8, 5.2)
plt.xlabel('Observer 1 Ratings')
plt.ylabel('Observer 2 Ratings')
plt.text(xlim[1] - 0.2 * (xlim[1] - xlim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0]),f'{corr:.3f}')

plt.subplot(2, 2, 4).set_title('Heatmap of Rating Agreement')
heatmap_data = pd.crosstab(db['Likert 2'].iloc[1::2], db['Likert 1'].iloc[1::2]).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)  # Create a crosstab for the heatmap
sns.heatmap(heatmap_data, annot=True, cmap='Reds', cbar=True, fmt='d')
plt.xlabel('Observer 1 Ratings')
plt.ylabel('Observer 2 Ratings')
plt.tight_layout()

plt.gca().invert_yaxis()  # Inverts the Y-axis


plt.figure('General correlation')
plt.subplot(2,3,1)
corr_matrix = db[['Likert 1', 'Likert 2', 'DICE']].iloc[1::2].corr()
plt.title("Correlation Primary")
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
custom_labels = ['Observer 1', 'Observer 2', 'DICE']  # Custom labels for x and y axis
heatmap.set_xticklabels(custom_labels, rotation=45, horizontalalignment='right')  # Custom x-axis labels with rotation
heatmap.set_yticklabels(custom_labels, rotation=0)

plt.subplot(2,3,4)
corr_matrix = db[['Likert 1', 'Likert 2', 'DICE']].iloc[::2].corr()
plt.title("Correlation Nodal")
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
custom_labels = ['Observer 1', 'Observer 2', 'DICE']  # Custom labels for x and y axis
heatmap.set_xticklabels(custom_labels, rotation=45, horizontalalignment='right')  # Custom x-axis labels with rotation
heatmap.set_yticklabels(custom_labels, rotation=0)

plt.subplot(2,3,2)
corr_matrix = db2[['Likert 1', 'Likert 2', 'DICE']].iloc[1::2].corr()
plt.title("Correlation Primary Perturbed")
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
custom_labels = ['Observer 1', 'Observer 2', 'DICE']  # Custom labels for x and y axis
heatmap.set_xticklabels(custom_labels, rotation=45, horizontalalignment='right')  # Custom x-axis labels with rotation
heatmap.set_yticklabels(custom_labels, rotation=0)

plt.subplot(2,3,5)
corr_matrix = db2[['Likert 1', 'Likert 2', 'DICE']].iloc[::2].corr()
plt.title("Correlation Nodal Perturbed")
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
custom_labels = ['Observer 1', 'Observer 2', 'DICE']  # Custom labels for x and y axis
heatmap.set_xticklabels(custom_labels, rotation=45, horizontalalignment='right')  # Custom x-axis labels with rotation
heatmap.set_yticklabels(custom_labels, rotation=0)


observer1 = 'Likert 1'
observer2 = 'Likert 2'

plt.subplot(2,3,3)
heatmap_data_obs1 = pd.crosstab(index=[db[observer1].iloc[1::2]], columns=["Count"]).reindex(range(1, 6), fill_value=0)
heatmap_data_obs2 = pd.crosstab(index=db[observer2].iloc[1::2], columns="Count").reindex(range(1, 6), fill_value=0)
# Concatenate the two heatmap datasets side by side
combined_heatmap_data = pd.concat([heatmap_data_obs1, heatmap_data_obs2], axis=1)
combined_heatmap_data.columns = ['Observer 1', 'Observer 2']  # Rename columns for clarity
# Create a single heatmap
sns.heatmap(combined_heatmap_data, annot=True, cmap='Reds', cbar=True, fmt='d')
# Add title and labels
plt.title('Grades Primary')
plt.xlabel('')
plt.ylabel('')
plt.xticks(ticks=[0.5, 1.5], labels=['Observer 1', 'Observer 2'], rotation=45)  # Center the ticks on the heatmap

plt.subplot(2,3,6)
heatmap_data_obs1 = pd.crosstab(index=[db[observer1].iloc[::2]], columns=["Count"]).reindex(range(1, 6), fill_value=0)
heatmap_data_obs2 = pd.crosstab(index=db[observer2].iloc[::2], columns="Count").reindex(range(1, 6), fill_value=0)
# Concatenate the two heatmap datasets side by side
combined_heatmap_data = pd.concat([heatmap_data_obs1, heatmap_data_obs2], axis=1)
combined_heatmap_data.columns = ['Observer 1', 'Observer 2']  # Rename columns for clarity
# Create a single heatmap
sns.heatmap(combined_heatmap_data, annot=True, cmap='Reds', cbar=True, fmt='d')
# Add title and labels
plt.title('Grades Nodal')
plt.xlabel('')
plt.ylabel('')
plt.xticks(ticks=[0.5, 1.5], labels=['Observer 1', 'Observer 2'], rotation=45)  # Center the ticks on the heatmap


plt.tight_layout()

# Show the plot
plt.show()