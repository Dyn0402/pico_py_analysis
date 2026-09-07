#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 05 10:22 PM 2025
Created in PyCharm
Created as pico_py_analysis/resistive_spread_sim.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def hex_mask(grid_size, center, radius):
    """Create a hexagonal mask within a square grid."""
    y, x = np.indices((grid_size, grid_size))
    cx, cy = center
    dx = np.abs(x - cx) / radius
    dy = np.abs(y - cy) / radius

    # Approximate hexagon equation: |x| + sqrt(3)*|y| <= 1
    return (dx + np.sqrt(3) * dy <= 1)


def main():
    # Simulation parameters
    grid_size = 1000  # 100x100 grid
    dx = 1e-3  # meters
    dy = dx
    dt = 1e-6  # seconds
    diffusion_const = 1e-1  # m^2/s

    steps = 5000
    plot_interval = 50

    # Editable center position (y, x)
    center = (500, 500)  # (row, column) -> (y, x)
    radius = 100  # in grid units

    # Create hexagonal mask
    mask = hex_mask(grid_size, center=center[::-1], radius=radius)

    # Initialize charge distribution
    q = np.zeros((grid_size, grid_size))
    q_new = np.zeros_like(q)

    # Deposit initial charge at center
    q[center] = 100.0 / (dx * dy)  # charge density in C/m^2

    def laplacian(z):
        return (
                -4 * z
                + np.roll(z, 1, axis=0)
                + np.roll(z, -1, axis=0)
                + np.roll(z, 1, axis=1)
                + np.roll(z, -1, axis=1)
        ) / dx ** 2

    # Store frames for plotting
    frames = []

    for t in range(steps):
        q_new = q + dt * diffusion_const * laplacian(q)

        # Enforce hex boundary: zero outside the mask
        q_new[~mask] = 0

        q = q_new.copy()

        if t % plot_interval == 0:
            frames.append(q.copy())

    # Plot time evolution
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [0, grid_size * dx, 0, grid_size * dy]
    im = ax.imshow(frames[0], origin='lower', cmap='hot', extent=extent, vmax=5000)
    ax.plot(center[1] * dx, center[0] * dy, 'bx', markersize=8, label='Initial Charge')
    ax.set_title("Charge Distribution Over Time")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Charge density (C/m²)")
    ax.legend()

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
