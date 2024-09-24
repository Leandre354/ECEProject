import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

Modality = 'CT' #CT or PET
Label = 'Nodal' #Nodal or Primary
# File paths
test = f'Analysis/{Modality}'

files = {
    'BASE': f'Inference/Image200/Analysis/Baseline/results_pymia.csv',
    'BLUR': [f'Inference/Image200/{test}/Blur1/results_Blur1_pymia.csv',
             f'Inference/Image200/{test}/Blur2/results_Blur2_pymia.csv',
             f'Inference/Image200/{test}/Blur3/results_Blur3_pymia.csv'],
    'NOISE': [f'Inference/Image200/{test}/Noise1/results_Noise1_pymia.csv',
              f'Inference/Image200/{test}/Noise2/results_Noise2_pymia.csv',
              f'Inference/Image200/{test}/Noise3/results_Noise3_pymia.csv'],
    'GHOST': [f'Inference/Image200/{test}/Ghost1/results_Ghost1_pymia.csv',
              f'Inference/Image200/{test}/Ghost2/results_Ghost2_pymia.csv',
              f'Inference/Image200/{test}/Ghost3/results_Ghost3_pymia.csv'],
    'SPIKE': [f'Inference/Image200/{test}/Spike1/results_Spike1_pymia.csv',
              f'Inference/Image200/{test}/Spike2/results_Spike2_pymia.csv',
              f'Inference/Image200/{test}/Spike3/results_Spike3_pymia.csv'],
    'BIAS': [f'Inference/Image200/{test}/Bias1/results_Bias1_pymia.csv',
             f'Inference/Image200/{test}/Bias2/results_Bias2_pymia.csv',
             f'Inference/Image200/{test}/Bias3/results_Bias3_pymia.csv'],
    'MOTION': [f'Inference/Image200/{test}/Motion1/results_Motion1_pymia.csv',
               f'Inference/Image200/{test}/Motion2/results_Motion2_pymia.csv',
               f'Inference/Image200/{test}/Motion3/results_Motion3_pymia.csv']
}

# Load the data into a dictionary
data = {}

# Loop over the file paths, concatenating them if there are multiple for a key
for name, paths in files.items():
    if isinstance(paths, list):  # If there are multiple files for this key
        data[name] = [pd.read_csv(path, delimiter=';') for path in paths]
    else:  # Single file
        data[name] = pd.read_csv(paths, delimiter=';')

# Define the metrics to plot
metrics = ['DICE', 'HDRFDST', 'SNSVTY', 'SPCFTY', 'ACURCY']
titles = ['DICE', 'Hausdorff', 'Sensitivity', 'Specificity', 'Accuracy']
metrics2 = ['DICE', 'DICE', 'DICE', 'DICE','DICE'] #, 'DICE', 'DICE', 'DICE','DICE', 'DICE', 'DICE', 'DICE','DICE']
titles2 = ['Volume','Surface Area','Compactness', 'Distance from Center','Boundary Length','CT Intensity Var',
            'PET Intensity Var','CT F/B Contrast','PET F/B Contrast',
            'SUVmax','CT Number (HU)', 'Regions','Entropy']




# Create a figure
fig, axes = plt.subplots(len(metrics), len(files), figsize=(15, 15), constrained_layout=True, num=f'{Modality} - {Label}')
fig2, axes2 = plt.subplots(3, len(files), figsize=(15, 15), constrained_layout=True, num=f'{Modality} - {Label} - Properties - 1')
fig3, axes3 = plt.subplots(3, len(files), figsize=(15, 15), constrained_layout=True, num=f'{Modality} - {Label} - Properties - 2')
fig4, axes4 = plt.subplots(3, len(files), figsize=(15, 15), constrained_layout=True, num=f'{Modality} - {Label} - Properties - 3')
fig5, axes5 = plt.subplots(4, len(files), figsize=(15, 15), constrained_layout=True, num=f'{Modality} - {Label} - Properties - 4')

# Function to plot boxplot and add text annotations
def plot_box(ax, df,db, metric, title, name):
    if name == 'BASE':
        data = df[metric]
        if Label == 'Nodal':
            data = data.iloc[::2]
        if Label == 'Primary':
            data = data.iloc[1::2]
    else:
        data = db[metric] - df[0][metric]
        data2 = db[metric] - df[1][metric]
        data3 = db[metric] - df[2][metric]
        if Label == 'Nodal':
            data = data.iloc[::2]
            data2 = data2.iloc[::2]
            data3 = data3.iloc[::2]
        if Label == 'Primary':
            data = data.iloc[1::2]
            data2 = data2.iloc[1::2]
            data3 = data3.iloc[1::2]

    data_replaced = np.where(np.isfinite(data), data, 0)
    if name != 'BASE':
        data_replaced2 = np.where(np.isfinite(data2), data2, 0)
        data_replaced3 = np.where(np.isfinite(data3), data3, 0)

    if name == 'BASE':
        ax.boxplot(data_replaced, flierprops=dict(marker='.', alpha=0.3))
    else:
        ax.boxplot([data_replaced, data_replaced2, data_replaced3], flierprops=dict(marker='.', alpha=0.3))
    if title == 'DICE':
        if name == 'BASE':
            ax.set_title(r"$\mathbf{" + name + "}$" + f"\n\n{title}")
        else:
            ax.set_title(r"$\mathbf{" + name + "}$" + f"\n\nΔ{title}")
    else:
        if name == 'BASE':
            ax.set_title(f'{title}')
        else:
            ax.set_title(f'Δ{title}')

def plot_box2(ax, df, db, dba, metric, title, name):
    if title == titles3[0]:
        if name == 'BASE':
            ax.set_title(r"$\mathbf{" + name + "}$" + f"\n\ndice v {title}", fontsize=7)
        else:
            ax.set_title(r"$\mathbf{" + name + "}$" + f"\n\nΔdice v {title}", fontsize=7)
    else:
        if name == 'BASE':
            ax.set_title(f'dice v {title}', fontsize=7)
        else:
            ax.set_title(f'Δdice v {title}', fontsize=7)

    if name == 'BASE':
        if Label == 'Nodal':
            ax.scatter(dba[title].iloc[::2], df[metric].iloc[::2], marker='.')

        if Label == 'Primary':
            ax.scatter(dba[title].iloc[1::2], df[metric].iloc[1::2], marker='.')
    else:
        x = dba[title]
        y0 = (db[metric] - df[0][metric])
        y1 = (db[metric] - df[1][metric])
        y2 = (db[metric] - df[2][metric])

        if Label == 'Nodal':
            x = x.iloc[::2]
            y0 = y0.iloc[::2]
            y1 = y1.iloc[::2]
            y2 = y2.iloc[::2]

        if Label == 'Primary':
            x = x.iloc[1::2]
            y0 = y0.iloc[1::2]
            y1 = y1.iloc[1::2]
            y2 = y2.iloc[1::2]

        x = np.where(np.isfinite(x), x, 0)

        mask = x != 0
        x = x[mask]
        y0 = y0[mask]
        y1 = y1[mask]
        y2 = y2[mask]

        pearson_corr0, _ = stats.pearsonr(x, y0)
        spearman_corr0, _ = stats.spearmanr(x, y0)
        kendall_corr0, _ = stats.kendalltau(x, y0)
        pearson_corr1, _ = stats.pearsonr(x, y1)
        spearman_corr1, _ = stats.spearmanr(x, y1)
        kendall_corr1, _ = stats.kendalltau(x, y1)
        pearson_corr2, _ = stats.pearsonr(x, y2)
        spearman_corr2, _ = stats.spearmanr(x, y2)
        kendall_corr2, _ = stats.kendalltau(x, y2)
        textstr = (f'1st : {pearson_corr0:.2f}\n' #S: {spearman_corr0:.2f} K: {kendall_corr0:.2f}\n'
                   f'2nd : {pearson_corr1:.2f}\n' #S: {spearman_corr1:.2f} K: {kendall_corr1:.2f}\n'
                   f'3rd : {pearson_corr2:.2f}') #S: {spearman_corr2:.2f} K: {kendall_corr2:.2f}')
        ax.scatter(x, y0, marker='.', color='green', label='1st')
        ax.scatter(x, y1, marker='+', color='blue', alpha=0.4, label='2nd')
        ax.scatter(x, y2, marker='x', color='red', alpha=0.2, label='3rd')
        ax.legend(loc='upper right', fontsize=6)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, 0.22, textstr, transform=ax.transAxes, fontsize=6, verticalalignment='top', bbox=props)

index = 0
index2 = 0
index3 = 0
index4 = 0
index5 = 0
# Plotting all subplots
for name, df in data.items():
    col = index // len(metrics)
    col2 = index2 // 3
    col3 = index3 // 3
    col4 = index4 // 3
    col5 = index5 // 4

    db = pd.read_csv('Inference/Image200/Analysis/Baseline/results_pymia.csv', delimiter=';')
    dba = pd.read_csv('Inference/Image200/Analysis/Baseline/results_add.csv', delimiter=',')

    metrics3 = metrics2[0:3]
    titles3 = titles2[0:3]

    for metric, title in zip(metrics3, titles3):
        row2 = index2 % len(metrics3)
        ax = axes2[row2, col2]
        plot_box2(ax, df, db, dba, metric, title, name)
        index2 += 1

    metrics3 = metrics2[0:3]
    titles3 = titles2[3:6]

    for metric, title in zip(metrics3, titles3):
        row3 = index3 % len(metrics3)
        ax = axes3[row3, col3]
        plot_box2(ax, df, db, dba, metric, title, name)
        index3 += 1

    metrics3 = metrics2[0:3]
    titles3 = titles2[6:9]

    for metric, title in zip(metrics3, titles3):
        row4 = index4 % len(metrics3)
        ax = axes4[row4, col4]
        plot_box2(ax, df, db, dba, metric, title, name)
        index4 += 1

    metrics3 = metrics2[0:4]
    titles3 = titles2[9:13]

    for metric, title in zip(metrics3, titles3):
        row5 = index5 % len(metrics3)
        ax = axes5[row5, col5]
        plot_box2(ax, df, db, dba, metric, title, name)
        index5 += 1

    for metric, title in zip(metrics, titles):
        row = index % len(metrics)
        ax = axes[row, col]
        plot_box(ax, df, db, metric, title, name)
        index += 1

plt.show()