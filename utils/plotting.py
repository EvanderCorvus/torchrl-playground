import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_paths(paths, v_c, char_length, size):
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    goal = patches.Rectangle((0.5-0.05, -0.05), 0.1, 0.1, edgecolor='black', facecolor='red')
    ax.add_patch(goal)
    ax2 = ax.twinx()
    for i in range(len(paths.files)):
        path = paths[f'arr_{i}']
        ax.plot(path[:, 0], path[:, 1], label=f'Path {i}', c='black', alpha=0.2)

    x = np.linspace(-size, size, 100)
    y = v_c*(1-(size*x)**2)
    ax2.plot(x, y, label='Velocity Profile', color='red', linestyle='--')
    ax2.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

    plt.title(f'$v_c = {v_c}, \sqrt{{D_RL/v_0}} = {char_length}$')
    plt.show()


def plot_flowfield(func, size, density, ax, **kwargs):
    x = np.linspace(-size, size, density)
    y = np.linspace(-size, size, density)
    X, Y = np.meshgrid(x, y)

    # Initialize u and v arrays
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u, v = func(X[i, j], Y[i, j], **kwargs)
            U[i, j] = u
            V[i, j] = v
    ax.streamplot(X, Y, U, V, density=2, linewidth=0.5)
