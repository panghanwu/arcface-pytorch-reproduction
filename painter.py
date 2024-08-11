import matplotlib.pyplot as plt
import numpy as np


def plot_3d_embeddings(embeddings, labels, centers=None, show=True, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = 0.97

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='white', linewidth=0, alpha=0.3)

    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='rainbow', s=15)

    if centers is not None:
        clabels = np.arange(len(centers))
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=clabels, cmap='rainbow', marker='*', s=50)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()