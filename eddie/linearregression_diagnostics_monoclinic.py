'''linearregression_diagnostics_monoclinic
Fit linear regression model and perform diagnostics
Only for monoclinic lattice parameters
'''

import csv
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


### INPUT PARAMETERS ##################################################
filename = 'Lattice parameters_monoclinic.csv'

# specify outlier points to exclude from analysis
#removeindex = [113,114,115,68,69]     # specify indices of rows to remove
removeindex = []     # specify indices of rows to remove

features = [0,1,2,3,4,5,6,7]  # features (column index) to consider

resp = -1   # which response variable? -4=a, -3=b, -2=c, -1=beta
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

X = data[:,-12:-4]  # features
header2_X = header2[-13:-5]  # headers of features
y = data[:,resp]   # lattice param

X = X[:,features]  # selected features
header2_X = [header2_X[i] for i in features]

k = X.shape[1]  # dimensions
n = X.shape[0]  # number of data points

print('Removed %g outliers. %g data points remaining.\n' % (len(removeindex),n))


# fit linear model
regr = linear_model.LinearRegression()
regr.fit(X, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# The mean square error
y_hat = regr.predict(X)
mse = np.mean((y_hat - y)**2)
msestat = mse*n/(n-k-1)
print('RMSE (ML, %g df) = %.4f' % (n,mse**0.5))   # ML community
print('RMSE (stats, %g df) = %.4f' % (n-k-1,msestat**0.5))   # stats community

# Coefficient of determination R^2 (multiple, not adjusted)
R2 = regr.score(X, y)
print('R^2 = %.4f' % R2)

# manual
y_bar = np.mean(y)
mst = np.mean((y_bar - y)**2)
##R2manual = 1-mse/mst

# adjusted R^2
R2adj = 1-mse*n*(n-1)/(mst*n*(n-k-1))
print('R^2 (adj) = %.4f' % R2adj)


# residuals
residuals = y_hat-y     # residuals
print('Residuals:')
print('  min = %.2f' % np.min(residuals))
print('  25%% = %.2f' % np.percentile(residuals,25))
print('  median = %.2f' % np.percentile(residuals,50))
print('  75%% = %.2f' % np.percentile(residuals,75))
print('  max = %.2f' % np.max(residuals))



### PLOTS
# plot y vs yhat
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(y, y_hat, color='black')
line = np.array([np.min(y), np.max(y)])     # line x=y (perfect fit)
plt.plot(line, line, color='blue', linewidth=3)
plt.ylabel('Predicted value (A)', fontsize=20)
plt.xlabel('Actual value (A)', fontsize=20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

# calculate leverage and std residuals
X2 = np.concatenate((np.ones((n,1)),X),axis=1)
leverage = np.diag(X2@np.linalg.inv(X2.T@X2)@X2.T)  # H_ii (hat matrix)
    # Note: np.std uses df=n
    # (np.sum((residuals-np.mean(residuals))**2)/(n))**.5
std = (np.sum((residuals-np.mean(residuals))**2)/(n-k-1))**.5
residuals_std = np.squeeze(residuals)/((1-leverage)**0.5*std);     # standardized residuals


# plot residuals vs each predictor
for i in range(X.shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X[:,i], residuals)
    bottomx, topx = plt.xlim()    # get axis limits
    line = np.array([np.min(y), np.max(y)])     # line x=y (perfect fit)
    plt.plot([bottomx, topx], [0, 0], color='black', linewidth=1,linestyle='--')
    plt.xlabel(header2_X[i], fontsize=20)
    plt.ylabel('Error (A)', fontsize=20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    

# standardized residuals normal q-q plot
fig = plt.figure()
ax = fig.add_subplot(111)
res = stats.probplot(np.squeeze(residuals_std), plot=ax)
plt.xlabel('Theoretical quantiles', fontsize=20)
plt.ylabel('Standardized residuals', fontsize=20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
# flag |std residuals| > 2
residuals_std_th_sort = res[0][0]    # sorted theoretical std residuals
residuals_std_sortindex = np.argsort(residuals_std,axis=0)  # indices of sorted actual std residuals

residuals_std_th = 10*np.ones(residuals_std_th_sort.shape)   # initialize
for i, t in enumerate(residuals_std_sortindex):
    residuals_std_th[t] = residuals_std_th_sort[i]    # unsorted theoretical std residuals


print('\n===Points flagged with |std residual| > 2:================')
print('index residual         %s    %s' % (header2[resp-1],header2[-1]))
index = [i for i, x in enumerate(residuals_std) if (x>2 or x<-2)]    # get indices
for i, txt in enumerate(index):
    ax.annotate(txt, (residuals_std_th[txt], residuals_std[txt]))
    print('%5.0f %8.4f  %8.4f    %s' % (txt,residuals[txt],data[txt,resp],ref[txt]))


# plot residuals vs yhat
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(y_hat, residuals)
plt.xlabel('Fitted values', fontsize=20)
plt.ylabel('Residuals', fontsize=20)
for i, txt in enumerate(index):
    ax.annotate(txt, (y_hat[txt], residuals[txt]))  # flag std |residuals| > 2
bottomx, topx = plt.xlim()    # get axis limits
plt.plot([bottomx, topx], [0, 0], color='black', linewidth=1,linestyle='--')
plt.xlim(bottomx, topx)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)


# plot std residuals vs leverage
h_flag = 2*(k+1)/n

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(leverage, residuals_std)
plt.xlabel('Leverage', fontsize=20)
plt.ylabel('Standardized residuals', fontsize=20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

print('\n===Points flagged with leverage > %.4f:================' % h_flag)
print('index residual         %s    %s' % (header2[resp-1],header2[-1]))
index2 = [i for i, x in enumerate(leverage) if x>h_flag]    # get indices
for i, txt in enumerate(index2):
    ax.annotate(txt, (leverage[txt], residuals_std[txt]))  # flag high leverage
    print('%5.0f %8.4f  %8.4f    %s' % (txt,residuals[txt],data[txt,resp],ref[txt]))

# Cook's distance contours
bottomx, topx = plt.xlim()    # get axis limits
bottomy, topy = plt.ylim()
maxy = np.max([np.abs(bottomy),np.abs(topy)])

D = 0.5;  # specify contour value
cook_h = np.linspace(D*(k+1)/(maxy**2+D*(k+1)),topx,100)   # vector of h for plotting contours
cook_e1 = (D*(k+1)*(1-cook_h)/cook_h)**0.5
cook_e2 = -(D*(k+1)*(1-cook_h)/cook_h)**0.5

plt.plot(cook_h, cook_e1, color='red', linewidth=1,linestyle='--')
plt.plot(cook_h, cook_e2, color='red', linewidth=1,linestyle='--')
plt.xlim(bottomx, topx)
plt.ylim(bottomy, topy)



plt.show()

