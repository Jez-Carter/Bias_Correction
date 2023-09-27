# %% Importing Packages
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Loading Data
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()
scenario_ampledata = np.load(f'{inpath}scenario_ampledata.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d.npy',allow_pickle='TRUE').item()

scenarios = [scenario_base,scenario_ampledata,scenario_2d]

# %% Defining functions to compute nearest points between outputs
def find_nearest_1d(point,array):
    index = (np.abs(array - point)).argmin()
    return(index)

def find_nearest_2d(point,array):
    distance = (array[:,0]-point[0])**2 + (array[:,1]-point[1])**2
    index = distance.argmin()
    return(index)

# %% Finding value to nearest climate model grid cell to observations
for scenario in scenarios:
    nearest_indecies = []
    ox = scenario['ox']
    cx = scenario['cx']
    for point in ox:
        try:
            nearest_indecies.append(find_nearest_2d(point,cx))
        except:
            nearest_indecies.append(find_nearest_1d(point,cx))
    scenario['cdata_nearest'] = np.array(scenario['cdata'])[nearest_indecies]

# %% Scatter plots showing correlation

fig, axs = plt.subplots(3, 2, figsize=(10, 12),dpi=600)

for axs,scenario in zip(axs,scenarios):
    axs[0].scatter(scenario['odata'],scenario['cdata_nearest'])
    axs[0].set_xlabel('Observational Data Values')
    axs[0].set_ylabel('Nearest Grid Cell Climate Model Values')
    axs[0].set_title('Correlation between Climate Model and Observations')
    axs[0].plot(np.arange(-1,4,0.5),np.arange(-1,4,0.5),linestyle='--')

    axs[1].scatter(scenario['odata'],scenario['cdata_nearest']-scenario['odata'])
    axs[1].set_xlabel('Observational Data Values')
    axs[1].set_ylabel('Difference in Values (nearest climate model to observation)')
    axs[1].set_title('Correlation between Bias and Observations')

plt.tight_layout()
