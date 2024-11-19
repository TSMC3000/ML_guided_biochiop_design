import pandas as pd
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE

## Create segmented cmaps
def seg_cmap(cmap_name='magma', value_range=(0.05, 0.9), num_colors=256, alpha=1.0, segments=2):
    
    try:
        colormap = plt.get_cmap(cmap_name)
    except ValueError:
        raise ValueError(f"Colormap '{cmap_name}' is not recognized by matplotlib.")
        
    values = np.linspace(value_range[0], value_range[1], num_colors)
    colors = colormap(values)
    colors[:, 3] = alpha
    
    seg_cmap = LinearSegmentedColormap.from_list(f'{cmap_name}_seg', colors, N=segments)
    
    return seg_cmap

## Parameters for scatter plot
def raw_scatter(df, features=['PHO (wt.%)', 'NGF (ng/mL)', 'LAM (ug/cm2)'], y_label='Neurite',vmin=0.0, vmax=1.0, marker='o', linewidths=0.25, s=50, alpha=1, elev=45, azim=30, edgecolor='k', cmap='viridis', show_cbar=True, cbar_label='Adhesion', cpad=0.1, xticks=[0.25, 0.50, 1.00, 1.50, 2.00], yticks=[0, 25, 50, 75, 100], zticks=[0, 41.3, 82.6, 124.0, 165.3], cbarticks=[0.25, 0.75], detatch=True, title='Raw Adhesion Status Data', labels=False, cbar_alpha=1.0):
    
    # Formatting matplotlib
    from mpl_toolkits.mplot3d.axis3d import Axis
    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs
        Axis._get_coord_info_old = Axis._get_coord_info  
        Axis._get_coord_info = _get_coord_info_new
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y, z, c = df[features[0]], df[features[1]], df[features[2]], df[y_label]
        
    sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s,  vmin=vmin, vmax=vmax, marker=marker, 
                    linewidths=linewidths, alpha=alpha, edgecolor=edgecolor)

    if show_cbar == True:
        cbar = plt.colorbar(sc, shrink=0.8, pad=cpad)
        cbar.ax.tick_params(labelsize=14, pad=7)
        cbar.set_ticks(cbarticks)
        cbar.solids.set_alpha(cbar_alpha)
        if labels:
            cbar.set_label(cbar_label, size=14, labelpad=10)
            
    if detatch == True:
        cbar.set_ticklabels(['Detached', 'Attached'], rotation=90, va='center')
        cbar.ax.tick_params(labelsize=12, size=0, pad=7)
        cbar.solids.set_alpha(cbar_alpha)
    
    plt.rcParams["font.family"] = "Arial"
    plt.title(title, fontsize=14)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.axes.set_xlim3d(left=min(xticks), right=max(xticks)) 
    ax.axes.set_ylim3d(bottom=max(yticks), top=min(yticks)) # Reverse y-axis
    ax.axes.set_zlim3d(bottom=min(zticks), top=max(zticks))
    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=5)
    ax.tick_params(axis='z', pad=5)
    ax.set_xticklabels(xticks, fontsize=14)
    ax.set_yticklabels(yticks, fontsize=14)
    
    ax.set_zticklabels(zticks, fontsize=14)
    if labels == True:
        ax.set_xlabel('Photoinitiator (wt.%)', fontsize=14, labelpad=10)
        ax.set_ylabel('NGF (ng/mL)', fontsize=14, labelpad=12)
        ax.set_zlabel(r"Peptide ($\mu g/cm^2$)", fontsize=14, labelpad=10)
    
    return sc

## Data Augmentation
def data_aug(data, features=['PHO_std', 'NGF_std', 'LAM_std'], 
             y_aug=False, y_label='MTT_std', ad_label='Adhesion', adhesion=True, num_aug=10,
             fnoise=0.05, ynoise=0.05, random_state=1):

    if random_state is not None:
        np.random.seed(random_state)
        
    missing_features = [feat for feat in features if feat not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in data columns: {missing_features}")   
    
    if y_aug:
        if y_label is None:
            raise ValueError("y_label must be provided when y_aug is True.")
        if y_label not in data.columns:
            raise ValueError(f"Variable '{y_label}' not found in data columns.")
            
    if adhesion:
        if ad_label not in data.columns:
            raise ValueError(f"Variable '{ad_label}' not found in data columns.")
            
    num_aug = num_aug
    data_aug = data.copy()

    for _ in range(num_aug):
        temp_aug = data.copy()
        
        for feature in features:
            mu = data[feature].values
            sigma = np.abs(mu*fnoise)
            aug_x = np.random.normal(loc=mu, scale=sigma)
            temp_aug[feature] = aug_x
            
            if adhesion:
                data['Adhesion'] = data['Adhesion']
            
        if y_aug:
            mu_y = data[y_label].values
            sigma_y = np.abs(mu_y * ynoise)
            aug_y = np.random.normal(loc=mu_y, scale=sigma_y)
            temp_aug[y_label] = aug_y

        data_aug = pd.concat([data_aug, temp_aug], ignore_index=True)
    
    return data_aug

# Models and parameters for Ensemble ANN
weak_idx = [0, 1, 2, 3, 4]
mid_idx = [5, 6, 7, 8, 9]
strong_idx = [10, 11, 12, 13, 14]
best_idx = [0, 1, 4, 7, 14] # Best

def select_ensemble(X, y, idx=best_idx):
    models = [
        ## Weak models
        # r2 = 0.1
        # Model #1117:
        #   - CV Metrics: Val R2 = 0.1037, Val MAE = 0.1465;
        #   - Train Metrics: R2 = 0.1120, MAE = 0.2064;
        MLPRegressor(activation='tanh', alpha=4.069179548172916e-06, batch_size=64,
                     hidden_layer_sizes=[5, 5, 5, 5, 5, 5, 5, 5, 4, 4],
                     learning_rate_init=1.4229520275134514e-05, max_iter=50000,
                     random_state=0, tol=1e-05),

        # Model #1926:
        #   - CV Metrics: Val R2 = 0.1124, Val MAE = 0.1675;
        #   - Train Metrics: R2 = 0.1268, MAE = 0.2089;
        MLPRegressor(activation='tanh', alpha=9.368302866664674e-05, batch_size=24,
                     hidden_layer_sizes=[5, 1],
                     learning_rate_init=2.683342414372862e-05, max_iter=50000,
                     random_state=6, tol=1e-05),

        # Model #2052:
        #   - CV Metrics: Val R2 = 0.1133, Val MAE = 0.1574;
        #   - Train Metrics: R2 = 0.1333, MAE = 0.2047;
        MLPRegressor(activation='tanh', alpha=1.0636353020794172e-06, batch_size=24,
                     hidden_layer_sizes=[5, 2],
                     learning_rate_init=4.042940432014313e-05, max_iter=50000,
                     random_state=4, tol=1e-05),

        # Model #1857:
        #   - CV Metrics: Val R2 = 0.1243, Val MAE = 0.1438;
        #   - Train Metrics: R2 = 0.1333, MAE = 0.2023;
        MLPRegressor(activation='tanh', alpha=5.11397827693426e-06, batch_size=40,
                     hidden_layer_sizes=[7, 5, 3],
                     learning_rate_init=7.098088946744915e-05, max_iter=50000,
                     random_state=5, tol=1e-05),

        # Model #1639:
        #   - CV Metrics: Val R2 = 0.1247, Val MAE = 0.1724;
        #   - Train Metrics: R2 = 0.1401, MAE = 0.2055;
        MLPRegressor(activation='tanh', alpha=9.368302866664674e-05, batch_size=16,
                     hidden_layer_sizes=[4, 1],
                     learning_rate_init=4.730172996096315e-05, max_iter=50000,
                     random_state=9, tol=1e-05),

        ## r2 = 0.5
        # Model #163:
        #   - CV Metrics: Val R2 = 0.5705, Val MAE = 0.0892;
        #   - Train Metrics: R2 = 0.5006, MAE = 0.1400;
        MLPRegressor(activation='tanh', alpha=1.4699245951666415e-07, batch_size=24,
                     hidden_layer_sizes=[10, 8, 7, 5, 3],
                     learning_rate_init=0.0013507653838908675, max_iter=50000,
                     random_state=2, tol=1e-05),

        # Model #820:
        #   - CV Metrics: Val R2 = 0.7103, Val MAE = 0.0668;
        #   - Train Metrics: R2 = 0.5019, MAE = 0.1466;
        MLPRegressor(activation='tanh', alpha=4.1520846921736066e-06, batch_size=8,
                     hidden_layer_sizes=[6, 6, 5, 5, 5, 4, 4, 4, 3, 3],
                     learning_rate_init=0.0023914925232281707, max_iter=50000,
                     random_state=10, tol=1e-05),

        # Model #638:
        #   - CV Metrics: Val R2 = 0.5024, Val MAE = 0.1175;
        #   - Train Metrics: R2 = 0.5058, MAE = 0.1485;
        MLPRegressor(activation='tanh', alpha=4.3604728711610275e-07, batch_size=8,
                     hidden_layer_sizes=[19, 4],
                     learning_rate_init=0.00027627417698698043, max_iter=50000,
                     random_state=7, tol=1e-05),

        # Model #1966:
        #   - CV Metrics: Val R2 = 0.8636, Val MAE = 0.0517;
        #   - Train Metrics: R2 = 0.5089, MAE = 0.1455;
        MLPRegressor(activation='tanh', alpha=3.4298152582719893e-06, batch_size=12,
                     hidden_layer_sizes=[20, 8],
                     learning_rate_init=0.003993721484968005, max_iter=50000,
                     random_state=10, tol=1e-05),

        # Model #1804:
        #   - CV Metrics: Val R2 = 0.5533, Val MAE = 0.1071;
        #   - Train Metrics: R2 = 0.5103, MAE = 0.1457;
        MLPRegressor(activation='tanh', alpha=4.069179548172916e-06, batch_size=52,
                     hidden_layer_sizes=[4, 4], learning_rate_init=0.006388197965907791,
                     max_iter=50000, random_state=9, tol=1e-05),

        ## Strong models
        # Model #421:
        #   - CV Metrics: Val R2 = 0.9503, Val MAE = 0.0364;
        #   - Train Metrics: R2 = 0.9807, MAE = 0.0294;
        MLPRegressor(activation='tanh', alpha=4.069179548172916e-06, batch_size=28,
                     hidden_layer_sizes=[11, 9, 8, 6],
                     learning_rate_init=0.003993721484968005, max_iter=50000,
                     random_state=0, tol=1e-05),

        # Model #71:
        #   - CV Metrics: Val R2 = 0.9730, Val MAE = 0.0244;
        #   - Train Metrics: R2 = 0.9846, MAE = 0.0262;
        MLPRegressor(activation='tanh', alpha=1.061943207327572e-05, batch_size=36,
                     hidden_layer_sizes=[12, 12, 11, 11, 10, 10, 9, 9],
                     learning_rate_init=0.004877798264481361, max_iter=50000,
                     random_state=7, tol=1e-05),

        # Model #1274:
        #   - CV Metrics: Val R2 = 0.9372, Val MAE = 0.0297;
        #   - Train Metrics: R2 = 0.9864, MAE = 0.0234;
        MLPRegressor(activation='tanh', alpha=5.11397827693426e-06, batch_size=28,
                     hidden_layer_sizes=[25, 22, 19, 16, 13, 10, 7],
                     learning_rate_init=0.003993721484968005, max_iter=50000,
                     random_state=4, tol=1e-05),

        # Model #1082:
        #   - CV Metrics: Val R2 = 0.9447, Val MAE = 0.0345;
        #   - Train Metrics: R2 = 0.9843, MAE = 0.0261;
        MLPRegressor(activation='tanh', alpha=1.061943207327572e-05, batch_size=12,
                     hidden_layer_sizes=[23, 21, 19, 18, 16, 14, 12, 10, 9, 7],
                     learning_rate_init=0.0023914925232281707, max_iter=50000,
                     random_state=5, tol=1e-05),

        # Model #1167:
        #   - CV Metrics: Val R2 = 0.9629, Val MAE = 0.0309;
        #   - Train Metrics: R2 = 0.9833, MAE = 0.0278;
        MLPRegressor(activation='tanh', alpha=1.6299644793531056e-05, batch_size=8,
                     hidden_layer_sizes=[25, 25, 24, 24, 24, 24, 23, 23, 23, 22],
                     learning_rate_init=0.0012622161782621338, max_iter=50000,
                     random_state=0, tol=1e-05),
    ]

    selected_models = [models[i] for i in idx]
    
    return selected_models

# Ensemble ANN prediction
base_mlps = []
def MLPs_predict(X, models=base_mlps, var=False):

    predictions = np.array([model.predict(X) for model in models])
    y_pred_ensemble = np.mean(predictions, axis=0)
    y_pred_var = np.var(predictions, axis=0)
    
    if var == True:
        return y_pred_ensemble, y_pred_var
    else:
        return y_pred_ensemble

# Create 3D grid for prediction
x_cols_std = ['PHO_std', 'NGF_std', 'LAM_std']

def grid_3d(cols=x_cols_std , num_points=201):
    
    x, y, z = [np.linspace(0, 1.0, num=num_points) for _ in range(3)]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    gd = np.stack([xx, yy, zz], axis=-1)
    gd = gd.reshape(-1, 3)
    df_gd = pd.DataFrame(gd, columns=cols)
    return df_gd

# Spearmans' Figures Parameters
raw_x_cols = ['PHO (wt.%)', 'NGF (ng/mL)', 'LAM (ug/cm2)']
r_x_cols = raw_x_cols[::-1]
p_color  = '#1F4D7D'
ax_color = 'k'

p_min = 0.001
r_min = 0.45

n_xcols = len(r_x_cols)
x_ticks = np.linspace(1, n_xcols, n_xcols)
x_lo, x_hi = 0.5, n_xcols + 0.5
bar_height = 0.3

def style_corr_bar(ax, legend_ncol=1, ):
    ax.plot([0, 0], [-1, n_xcols + 1],  linestyle='-', color=ax_color, lw=0.5)
    ax.plot([r_min, r_min], [-1, n_xcols + 1],  linestyle='--', color='grey', lw=0.5)
    ax.plot([-r_min, -r_min], [-1, n_xcols + 1],  linestyle='--', color='grey', lw=0.5)
    ax.set_xlabel('spearman $r$')
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax.tick_params(axis='x', which='both')
    ax.set_xlim(-1, 1)
    ax.set_ylim(x_lo, x_hi)
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(r_x_cols)
    ax.minorticks_off()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=legend_ncol, fancybox=False, shadow=False)
    fig.tight_layout()