'''genericregression_diagnostics
Make diagnostic plots from any regression
Copy/paste yhat data in from R
Only works for tetragonal lattice parameters
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


### INPUT DATA ########################################################
# load X,y data from here
filename = 'Lattice parameters_tetragonal.csv'

# specify outlier points to exclude from analysis
#removeindex = [113,114,115,68,69]     # specify indices of rows to remove
removeindex = []     # specify indices of rows to remove

features = [1,5,7]

resp = -2   # which response variable? -2=a, -1=c

# input yhat data here
y_hat = np.array([5.152227,5.159255,5.152227,5.128840,5.158915,5.155165,5.151640,5.147859,5.146741,5.149078,5.151052,5.153969,5.156132,5.159633,5.160284,5.161825,5.166397,5.137821,5.139253,5.141206,5.146034,5.149209,5.153768,5.156167,5.171588,5.122029,5.128840,5.138501,5.138501,5.138501,5.138501,5.151986,5.136640,5.141210,5.153514,5.150407,5.161210,5.147185,5.157866,5.170169,5.148196,5.160215,5.174164,5.128677,5.141743,5.158960,5.173119,5.190240,5.146097,5.147114,5.148141,5.149178,5.151285,5.142114,5.141136,5.143886,5.144947,5.146020,5.147106,5.149320,5.155120,5.149320,5.143886,5.141801,5.140775,5.139759,5.138752,5.140910,5.141913,5.142930,5.145538,5.142930,5.140414,5.137982,5.135626,5.133337,5.132439,5.131993,5.131549,5.142690,5.143711,5.144748,5.145801,5.147412,5.144748,5.142185,5.139715,5.137329,5.135019,5.132776,5.132151,5.131377,5.141450,5.142444,5.143454,5.144479,5.145521,5.146580,5.147656,5.149305,5.146580,5.143964,5.141450,5.139029,5.136693,5.134432,5.132151,5.133776,5.132571,5.131722,5.130883,5.130466,5.130052,5.129640,5.129230,5.128821,5.128415,5.128010,5.127608,5.127207,5.126807,5.136071,5.136522,5.136975,5.137432,5.137892,5.138356,5.138822,5.139292,5.140243,5.141209,5.142190,5.143186,5.144199,5.145228,5.146275,5.147339,5.148423,5.149525,5.151215,5.148423,5.145749,5.143186,5.140724,5.138356,5.136071,5.133776,5.135377,5.134165,5.133313,5.132471,5.132054,5.131640,5.131228,5.130818,5.130411,5.130005,5.129602,5.129202,5.128803,5.128406,5.128011,5.127617,5.127226,5.126448,5.125676,5.124909,5.124148,5.126254,5.128406,5.130614,5.132890,5.135377,5.138501,5.136830,5.135547,5.134293,5.133066,5.131865,5.130688,5.129532,5.128397,5.127280,5.138501,5.104413,5.098091,5.098100,5.103318,5.105985,5.108786,5.113798,5.118658,5.120833,5.120753,5.123878,5.088634,5.097604,5.100019,5.102148,5.102088,5.102493,5.107643,5.118261,5.114627,5.120801,5.127284,5.097831,5.129008,5.138090,5.147577,5.157195,5.132752,5.141977,5.152043,5.162358,5.129654,5.130113,5.131027,5.132846,5.135105,5.137358,5.128271,5.125468,5.124521,5.124045,5.129869,5.130789,5.131706,5.125207,5.097831,5.095581,5.093307,5.091018,5.088724,5.086435,5.075411,5.065995,5.097831,5.095581,5.093307,5.091018,5.088724,5.086435,5.097831,5.111188,5.125006,5.138318,5.150147,5.097644,5.096853,5.095496,5.093609,5.091232,5.088724,5.079698,5.071382,5.064414,5.059435,5.095771,5.093691,5.091567,5.101074,5.104391,5.107768,5.111188,5.114636,5.097815,5.097644,5.097322,5.096853,5.096243,5.097831,5.095771,5.093691,5.091567,5.097831,5.101074,5.104391,5.107768,5.111188,5.114636,5.125006,5.138318,5.150147,5.159508,5.097815,5.097644,5.097322,5.096853,5.096243,5.095496,5.094616,5.093609,5.121419,5.128353,5.135044,5.117215,5.124323,5.105772,5.112895,5.120108,5.127274,5.101446,5.108507,5.115756,5.137694,5.147836,5.159149,5.133049,5.140298,5.147991,5.131711,5.138812,5.146199,5.137047,5.144602,5.152819,5.125460,5.141505,5.160355,5.079053,5.064880,5.043372,5.102766,5.104284,5.105163,5.113871,5.081079,5.104229,5.117542,5.128506,5.137668,5.081106,5.104147,5.117414,5.128358,5.137525,5.081126,5.104060,5.117279,5.128202,5.137375,5.081140,5.117137,5.128039,5.137218,5.097757,5.099853,5.102100,5.102370,5.104907,5.105781,5.107610,5.109890,5.110114,5.112028,5.112942,5.113217,5.115515,5.116506,5.118385,5.120740,5.120745,5.088935,5.088993,5.088990,5.088925,5.088798,5.101500,5.103764,5.106046,5.109938,5.110642,5.115241,5.115727,5.097051,5.101500,5.106046,5.110642,5.115241,5.099163,5.107642,5.115502,5.122040,5.101315,5.108130,5.115080,5.122029,5.128840,5.101315,5.108130,5.115080,5.122029,5.128840,5.097831,5.097450,5.097070,5.096567,5.102766,5.101430,5.100110,5.098807,5.102273,5.101751,5.101199,5.097487,5.097127,5.096752,5.103529,5.104607,5.105689,5.107863,5.110051,5.120634,5.123599,5.125958,5.128858,5.132638,5.136310,5.140358,5.143847,5.147658,5.151364,5.155071,5.159050,5.163377,5.179576,5.186146,5.193326,5.203125,5.087405,5.086996,5.085213,5.084282,5.088995,5.088786,5.088617,5.097715,5.101160,5.117597,5.096879,5.101107,5.105342,5.109546,5.113682,5.115008,5.160228,5.097051,5.101500,5.106046,5.110642,5.115241,5.119795,5.123691,5.125743,5.129754,5.131865,5.134047,5.137261,5.139326,5.140829,5.142189,5.142886,5.144786,5.145815,5.146735,5.147547,5.148250,5.148846,5.149335,5.149716,5.150025,5.150191,5.150220,5.150143,5.149961,5.149778,5.136467,5.137759,5.138985,5.140144,5.141468,5.142657,5.143890,5.145161,5.146381,5.147657,5.148882,5.150035,5.151395,5.152715,5.154006,5.155340,5.156763,5.158210,5.159735,5.161124,5.162493,5.163880,5.165217,5.165998,5.088993,5.088990,5.088925,5.088798,5.088610,5.088360,5.147570,5.144473,5.139188,5.134177,5.129453,5.134461,5.139706,5.145258,5.151184,5.157553,5.100550,5.091325,5.079608,5.097831,5.128490,5.137630,5.102766,5.132009,5.141267,5.149951,5.152004,5.154091,5.156220,5.158398,5.160630,5.162924,5.165286,5.167722,5.170240,5.172845,5.175439,5.102957,5.126483,5.137084,5.147920,5.154875,5.159810,5.162386,5.165044,5.167790,5.173573,5.183070,5.193715,5.201379,5.105689,5.107863,5.110051,5.115586,5.149982,5.151294,5.152644,5.154037,5.155475,5.159278,5.165084,5.169540,5.103413,5.104536,5.104536,5.103551])
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

X = X[:,features]  # selected features
header2_X = [header2_X[i] for i in features]

k = X.shape[1]  # dimensions
n = X.shape[0]  # number of data points

print('Removed %g outliers. %g data points remaining.\n' % (len(removeindex),n))




# The mean square error
mse = np.mean((y_hat - y)**2)
print('RMSE = %.6f' % mse**0.5)   # ML community

# Coefficient of determination R^2 (multiple, not adjusted)
y_bar = np.mean(y)
mst = np.mean((y_bar - y)**2)
R2 = 1-mse/mst
print('R^2 = %.4f' % R2)


# residuals
residuals = y_hat-y     # residuals
print('Residuals:')
print('  min = %.6f' % np.min(residuals))
print('  25%% = %.6f' % np.percentile(residuals,25))
print('  median = %.6f' % np.percentile(residuals,50))
print('  75%% = %.6f' % np.percentile(residuals,75))
print('  max = %.6f' % np.max(residuals))



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
print('index residual           %s    %s' % (header2[-2],header2[-1]))
index = [i for i, x in enumerate(residuals_std) if (x>2 or x<-2)]    # get indices
for i, txt in enumerate(index):
    ax.annotate(txt, (residuals_std_th[txt], residuals_std[txt]))
    print('%5.0f %8.2f     %8.2f    %s' % (txt,residuals[txt],data[txt,-1],ref[txt]))


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
print('index residual           %s    %s' % (header2[-2],header2[-1]))
index2 = [i for i, x in enumerate(leverage) if x>h_flag]    # get indices
for i, txt in enumerate(index2):
    ax.annotate(txt, (leverage[txt], residuals_std[txt]))  # flag high leverage
    print('%5.0f %8.2f     %8.2f    %s' % (txt,residuals[txt],data[txt,-1],ref[txt]))

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

