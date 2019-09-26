import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

category_names = ['error rate', 'accuracy']
results = {'Happy': [0.0030, 0.9970], 'Angry': [0.0031, 0.9969], 'Neutral': [0.0, 1.0], 'Sad': [0.0, 1.0]
           }
graph_title = 'my_graph_title'

def draw_scatter(h, a, n, s, title):
    happy_x, happy_y = h
    angry_x, angry_y = a
    neutral_x, neutral_y = n
    sad_x, sad_y = s
    
    sns.set(style="whitegrid")
    sns.residplot(happy_x, happy_y, color="g").set_title(title)
    sns.residplot(angry_x, angry_y, color="r").set_ylabel('score on the correct class')
    sns.residplot(neutral_x, neutral_y, color="y").set_xlabel('deamplified amount measured in decibels')
    sns.residplot(sad_x, sad_y, color="b")

def draw(results, category_names, graph_title):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.set_title(graph_title, y=-0.1)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'black' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c < 0.01:
                continue
            ax.text(x, y, str(float(c)), ha='center',
                    va='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax
