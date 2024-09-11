import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

Modality = 'PET' #Modality which is perturbed (CT or PET)
Label = 'Nodal' #Select label for the computation (Primary or Nodal)

#all csv files of each perturbation and each degree
files = {
    'BLUR1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Blur1/results_Blur1_pymia.csv', delimiter=';'),
    'BLUR2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Blur2/results_Blur2_pymia.csv', delimiter=';'),
    'BLUR3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Blur3/results_Blur3_pymia.csv', delimiter=';'),
    'NOISE1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Noise1/results_Noise1_pymia.csv', delimiter=';'),
    'NOISE2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Noise2/results_Noise2_pymia.csv', delimiter=';'),
    'NOISE3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Noise3/results_Noise3_pymia.csv', delimiter=';'),
    'GHOST1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Ghost1/results_Ghost1_pymia.csv', delimiter=';'),
    'GHOST2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Ghost2/results_Ghost2_pymia.csv', delimiter=';'),
    'GHOST3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Ghost3/results_Ghost3_pymia.csv', delimiter=';'),
    'SPIKE1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Spike1/results_Spike1_pymia.csv', delimiter=';'),
    'SPIKE2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Spike2/results_Spike2_pymia.csv', delimiter=';'),
    'SPIKE3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Spike3/results_Spike3_pymia.csv', delimiter=';'),
    'BIAS1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Bias1/results_Bias1_pymia.csv', delimiter=';'),
    'BIAS2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Bias2/results_Bias2_pymia.csv', delimiter=';'),
    'BIAS3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Bias3/results_Bias3_pymia.csv', delimiter=';'),
    'MOTION1': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Motion1/results_Motion1_pymia.csv', delimiter=';'),
    'MOTION2': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Motion2/results_Motion2_pymia.csv', delimiter=';'),
    'MOTION3': pd.read_csv(f'Inference/Image200/Analysis/{Modality}/Motion3/results_Motion3_pymia.csv', delimiter=';')
}

#Loading the metrics and properties of baseline
db = pd.read_csv('Inference/Image200/Analysis/Baseline/results_pymia.csv', delimiter=';')
dba = pd.read_csv('Inference/Image200/Analysis/Baseline/results_add.csv', delimiter=',')


correlation = []
correlation2d = []

#Properties used
titles = ['Volume', 'Surface Area', 'Compactness', 'Distance from Center', 'Boundary Length', 'CT Intensity Var',
            'PET Intensity Var', 'CT F/B Contrast', 'PET F/B Contrast',
            'SUVmax', 'CT Number (HU)', 'Regions', 'Entropy']

#Go through the files
for name, df in files.items():
    correlation = []
    #Go through the properties
    for title in titles:
        #Take only the Nodal label (even)
        if Label == 'Nodal':
            delta_dice = (db['DICE'] - df['DICE']).iloc[::2]
            volume = dba[title].iloc[::2]
        #Take only the Primary label (odd)
        if Label == 'Primary':
            delta_dice = (db['DICE'] - df['DICE']).iloc[1::2]
            volume = dba[title].iloc[1::2]
        #Remove the problematic values (inf, -inf, Nan)
        volume = np.where(np.isfinite(volume), volume, 0)
        delta_dice = np.array(delta_dice)

        #Compute the correlation with pearson
        corr, p = stats.pearsonr(volume, delta_dice)
        correlation.append(corr)

    correlation2d.append(correlation)

# Create a heatmap of correlation
plt.figure(figsize=(30, 8), num=f'{Modality} - {Label}')  # Adjust the figure size
plt.tight_layout()

sns.heatmap(correlation2d, annot=True, cmap='bwr', vmin=-1, vmax=1, square=True, cbar=True, fmt='.2f', xticklabels=titles,  # Column labels (metrics)
            yticklabels=files.keys(), annot_kws={"size": 6})

# Show the plot
plt.show()