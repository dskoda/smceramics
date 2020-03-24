'''linearregression_bestsubsets_cv_tetragonal
Perform best subsets with linear regression
Plot RMSE vs model size
RMSE calculated using K-fold CV
Only for tetragonal lattice parameters
'''

import csv
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
import time


startall = time.time()

### INPUT PARAMETERS ##################################################
filename = 'Lattice parameters_tetragonal.csv'

# specify outlier points to exclude from analysis
#removeindex = [113,114,115,68,69]     # specify indices of rows to remove
removeindex = []     # specify indices of rows to remove

modelsize = list(range(1,9))   # try models of these sizes, (1,10) tries all possible sizes
features = [0,1,2,3,4,5,6,7]  # features (column index) to consider

resp = -2   # which response variable? -2=a, -1=c

K = 10  # K-fold CV (567 for LOOCV)
#######################################################################

# read in headers
ref_all = []    # initialize list that contains ref names of data points
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            header = f'{", ".join(row)}'    # column headings
            header2 = row   # each name stored separately, indexable
            line_count += 1
        else:
            ref_all.append(row[-1])     # store ref name
            line_count += 1
    print(f'Processed {line_count-1} data points.\n')

# read in data
data_all = np.genfromtxt(filename, delimiter=',')
data_all = data_all[1:,:-1]     # chop off nan

# remove outliers and create feature/value arrays
ref = ref_all.copy()
for i in sorted(removeindex, reverse=True):
    del ref[i]
data = data_all.copy()
data = np.delete(data,removeindex,axis=0)

X = data[:,-10:-2]  # features
header2_X = header2[-11:-3]  # headers of features
y = data[:,resp]   # lattice param

k = X.shape[1]  # dimensions
n = X.shape[0]  # number of data points

print('Removed %g outliers. %g data points remaining.\n' % (len(removeindex),n))

# initialize arrays
mse = -100000*np.ones((int(comb(len(features),round(len(features)/2))),len(modelsize)))
    # each column corresponds to modelsize
msestat = mse.copy()
mse_best = np.zeros(len(modelsize))
mse_best_indices = mse_best.copy()

# define cross validation
cv = KFold(n_splits=K,shuffle=True,random_state=0)
regr = linear_model.LinearRegression()        # fit linear model

# loop through model sizes
for i in range(len(modelsize)):
    print('\nComputing model size = %g...' % modelsize[i])
    
    # figure out combinations
    indices = list(itertools.combinations(features,modelsize[i]))

    # loop through all combinations
    for j in range(len(indices)):
        X_temp = X[:,indices[j]]    # extract features

        MSE = 0    # initialize

        for train, test in cv.split(X_temp):
            X_train = X_temp[train]
            y_train = y[train]
            X_test = X_temp[test]
            y_test = y[test]

            regr.fit(X_train,y_train)    # fit model
            MSE += np.mean((regr.predict(X_test) - y_test)**2)   # add MSE contribution

        # compute average RMSE, fitstatus from K CV runs and store
        mse[j,i] = (MSE/K)**0.5


        if np.mod(j,100)==0:
            print('  %g of %g completed.' % (j, len(indices)))

    # find best model
    mse_best[i] = np.min(mse[:,i][mse[:,i]>0])
    mse_best_indices[i] = np.argmin(mse[:,i][mse[:,i]>0])

    # print best models
    print('\n  Best model:')
    print('   ',[header2_X[k] for k in indices[int(mse_best_indices[i])]])
    print('   ',[k for k in indices[int(mse_best_indices[i])]])
    print('    RMSE = %.4f\n' % mse_best[i])



# plot MSE
fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=.15, right=0.95, bottom=0.1, top=0.95)
for i in range(len(modelsize)):
    plt.scatter(modelsize[i]*np.ones(mse[:,i][mse[:,i]>0].shape), mse[:,i][mse[:,i]>0], color='C0')
plt.plot(modelsize, mse_best, marker='o', linestyle='-', color='r', linewidth=2)
plt.xlabel('Model size', fontsize=20); plt.ylabel('RMSE (A)', fontsize=20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
#plt.ylim(0, 1.05*np.max(mse))
plt.ylim(0, 1.05*np.max(mse[:,0]))
plt.xticks(modelsize, modelsize)



# plot best MSE only
fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=.15, right=0.95, bottom=0.1, top=0.95)
plt.plot(modelsize, mse_best, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Model size', fontsize=20); plt.ylabel('RMSE (A)', fontsize=20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
#plt.ylim(0, 1.05*np.max(mse_best))
plt.ylim(0, 1.05*mse_best[0])
plt.xticks(modelsize, modelsize)




print(mse_best)


endall = time.time()
print('\nTotal time elasped is %4f seconds.' % (endall - startall))



plt.show()
