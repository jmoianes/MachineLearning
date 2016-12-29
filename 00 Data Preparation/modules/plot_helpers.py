import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler

def plot_scaling(x, y, scalers=None, max_plot_columns=3):
    if scalers is None:
        print 'There are no scalers to plot'
        pass
    
    i = 0
    total_plots = len(scalers)+1
    if total_plots < max_plot_columns:
        columns = total_plots
        rows = 1
    else:
        columns = max_plot_columns
        rows = total_plots // max_plot_columns
    
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4))
    
    # original
    axes[i].scatter(x, y, c='r', label='original')
    axes[0].legend(loc='upper left')
    axes[0].set_title('original data', y=1.05)
    i += 1
    
    # scaling
    for scaler in scalers:
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        scaler.fit(y)
        y_scaled = scaler.transform(y)
        
        axes[i].scatter(x_scaled, y_scaled, c='b', label='feature scaling')
        axes[i].legend(loc='upper left')
        axes[i].set_title(scaler, y=1.05)
        i += 1
        
def plot_scaling_comparison(x, y):
    plt.figure(figsize=(15, 8))
    main_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

    main_ax.scatter(x, y, c=y)
    maxx = np.abs(x).max()
    maxy = np.abs(y).max()

    main_ax.set_xlim(-maxx - 1, maxx + 1)
    main_ax.set_ylim(-maxy - 1, maxy + 1)
    main_ax.set_title("Original Data", y=1.02)
    other_axes = [plt.subplot2grid((2, 4), (i, j)) for j in range(2, 4) for i in range(2)]

    for ax, scaler in zip(other_axes, [StandardScaler(), RobustScaler(),
                                       MinMaxScaler(), Normalizer(norm='l2')]):
        x_ = scaler.fit_transform(x)
        y_ = scaler.fit_transform(y)
        ax.scatter(x_, y_, c=y)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(type(scaler).__name__, y=1.02)

    other_axes.append(main_ax)

    for ax in other_axes:
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')