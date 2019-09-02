#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import matplotlib.patches as patches

colors = ['#ff0000', '#0000ff']
arrowp = '#cc0000'
arrown = '#0000cc'
arrowsize = 5
arrowsize2 = 3
mutation_scale = 50
spaceAbove = 0.20

markers = ["o", "D"]
matplotlib.rcParams.update({'font.size': 24})
rc('text', usetex=True)  # use same font as Latex
matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}'
    ]
plt.rcParams['svg.fonttype'] = 'none'  # selectable text in SVG


Y = np.array([2, 2, 2,
              1, 1, 1, 1])
X = np.array([[2, 6], [4, 2], [-1, -1],
              [0, 0.5], [5, 4.5], [2, 9], [9, 7]])


fig = plt.figure(1, figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.set_title("Similarity constraints")
for (i, c) in enumerate(sorted(np.unique(Y))):
    Xc = X[Y == c]
    ax.scatter(Xc[:, 0], Xc[:, 1], c=colors[i], edgecolor='black',
               linewidth='1', marker=markers[i], s=1000, zorder=2)

ax.add_patch(patches.Circle([5, 4.5], 3, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([-1, -1], 3, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([5, 4.5], 2.9, fill=True, linestyle='dashed',
                            color="grey", alpha=0.2))
ax.add_patch(patches.Circle([-1, -1], 2.9, fill=True, linestyle='dashed',
                            color="grey", alpha=0.2))
ax.text(6.3, 4.5+spaceAbove, "$1$")
ax.annotate("",
            xy=(5, 4.5), xycoords='data',
            xytext=(8, 4.5), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))
ax.text(0.3, -1+spaceAbove, "$1$")
ax.annotate("",
            xy=(-1, -1), xycoords='data',
            xytext=(2, -1), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))

ax.annotate("",
            xy=(-0.01, 1.31), xycoords='data',
            xytext=(2, 6), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrown,
                            lw=arrowsize))
ax.annotate("",
            xy=(1.15, 0.29), xycoords='data',
            xytext=(4, 2), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrown,
                            lw=arrowsize))


ax.annotate("",
            xy=(3.15, 3.02), xycoords='data',
            xytext=(0, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrowp,
                            lw=arrowsize))
ax.annotate("",
            xy=(3.62, 6.57), xycoords='data',
            xytext=(2, 9), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrowp,
                            lw=arrowsize))
ax.annotate("",
            xy=(7.12, 5.825), xycoords='data',
            xytext=(9, 7), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrowp,
                            lw=arrowsize))
ax.set_xlim([-2, 10])
ax.set_ylim([-2, 10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(1, 2, 2)
ax.set_title("Dissimilarity constraints")
for (i, c) in enumerate(sorted(np.unique(Y))):
    Xc = X[Y == c]
    ax.scatter(Xc[:, 0], Xc[:, 1], c=colors[i], edgecolor='black',
               linewidth='1', marker=markers[i], s=1000, zorder=2)

ax.add_patch(patches.Circle([5, 4.5], 4, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([5, 4.5], 3, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([-1, -1], 4, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([-1, -1], 3, fill=False, linestyle='dashed'))
ax.add_patch(patches.Circle([5, 4.5], 2.9, fill=True, linestyle='dashed',
                            color="grey", alpha=0.2))
ax.add_patch(patches.Circle([-1, -1], 2.9, fill=True, linestyle='dashed',
                            color="grey", alpha=0.2))
ax.text(6.3, 4.5+spaceAbove, "$1$")
ax.annotate("",
            xy=(5, 4.5), xycoords='data',
            xytext=(8, 4.5), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))
ax.text(8.1, 4.5+spaceAbove, "$m$")
ax.annotate("",
            xy=(8, 4.5), xycoords='data',
            xytext=(9, 4.5), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))
ax.text(0.3, -1+spaceAbove, "$1$")
ax.annotate("",
            xy=(-1, -1), xycoords='data',
            xytext=(2, -1), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))
ax.text(2.1, -1+spaceAbove, "$m$")
ax.annotate("",
            xy=(2, -1), xycoords='data',
            xytext=(3, -1), textcoords='data',
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,
                            mutation_scale=12, color="0.5", lw=arrowsize2))

ax.annotate("",
            xy=(1.45, 2.675), xycoords='data',
            xytext=(0, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrown,
                            lw=arrowsize))
ax.annotate("",
            xy=(0.98, 6.51), xycoords='data',
            xytext=(2, 6), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrowp,
                            lw=arrowsize))
ax.annotate("",
            xy=(3.35, 0.375), xycoords='data',
            xytext=(4, 2), textcoords='data',
            arrowprops=dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                            mutation_scale=mutation_scale, color=arrowp,
                            lw=arrowsize))

ax.set_xlim([-2, 10])
ax.set_ylim([-2, 10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(wspace=0)
plt.savefig("toyConstraint.pdf", bbox_inches='tight')
plt.show()
