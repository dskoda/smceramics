'''correlation_matrix
Plot correlation matrix and scatterplot matrix
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


### INPUT PARAMETERS ##################################################
filename = 'Lattice parameters_tetragonal.csv'
#filename = 'Lattice parameters_monoclinic.csv'

# which predictor variables to plot
pred = ['en_p','ea','valence','pettifor','rad_ionic','rad_slater','rad_clementi','T']  

# which response variable to plot
resp = ['a']    # tetragonal: 'a','c'. monoclinic: 'a','b','c',beta'

# specify outlier points to exclude from analysis
#removeindex = [113,114,115,68,69]     # specify indices of rows to remove
removeindex = []     # specify indices of rows to remove
#######################################################################


# read in data
data = pd.read_csv(filename,skiprows=removeindex, usecols=pred+resp)


# calculate the correlation matrix
plt.figure(figsize=(10,8))
sns.set(font_scale=2)   # set axis label font size
ax = sns.heatmap(data.corr(),annot=True, annot_kws={"size":13}, fmt=".2f", cmap='RdBu_r',
            vmin=-1, vmax=1, cbar_kws={'ticks': [-1, -0.5, 0, 0.5, 1], 'label': 'Pearson correlation coefficient'})
cbar_axes = ax.figure.axes[-1].yaxis.label.set_size(30)     # set colorbar label font size
#plt.savefig('correlationmatrix.png', bbox_inches='tight', dpi=300)


### scatterplot matrix
s1 = scatter_matrix(data, alpha=0.5, figsize=(12,12), diagonal='kde')
for ax in s1.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 13, rotation = 0)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 13, rotation = 90)
    plt.setp(ax.yaxis.get_majorticklabels(), 'size', 11) #y ticklabels
    plt.setp(ax.xaxis.get_majorticklabels(), 'size', 11) #x ticklabels
# fix the tick labels of the top-left plot (not sure why it's wonky)
new_labels = [round(float(i.get_text()), 2) for i in s1[0,0].get_yticklabels()]
s1[0,0].set_yticklabels(new_labels)


# rank predictors by R and print
rsort = np.abs(data.corr().iloc[-1]).sort_values()
print('Absolute value of correlation coefficient:')
print(rsort)


plt.show()
