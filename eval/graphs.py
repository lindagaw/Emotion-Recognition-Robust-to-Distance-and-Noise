import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def histogram(h, a, n, s, x_description, title):
    
    x_measurement = [2, 4, 6, 8]
    
    h_x, h_acc = sort_pair_ascending_xs(h[0], h[1])
    a_x, a_acc = sort_pair_ascending_xs(a[0], a[1])
    n_x, n_acc = sort_pair_ascending_xs(n[0], n[1])
    s_x, s_acc = sort_pair_ascending_xs(s[0], s[1])
    
    x = np.arange(len(x_measurement))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    
    rect1 = ax.bar(x - 2 * width, h_acc, width, label='Accuracy on Happy', color='#8dff33')
    rect2 = ax.bar(x - width, a_acc, width, label='Accuracy on Angry', color='#ff5733')
    rect3 = ax.bar(x, n_acc, width, label='Accuracy on Neutral', color='#ffd133')
    rect4 = ax.bar(x + width, s_acc, width, label='Accuracy on Sad', color='#3374ff')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title, y=-0.4)
    ax.set_ylabel('Accuracy')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(x_description)
    ax.legend()

def sort_pair_ascending_xs(xs, ys):
    li = []
    for x, y in zip(xs, ys):
        li.append((x, y))
    
    li = sorted(li, key=lambda x: x[0])
    
    ret_xs = []
    ret_ys = []
    for x, y in li:
        ret_xs.append(x)
        ret_ys.append(y)
        
    return ret_xs, ret_ys

def draw_scatter(h, a, n, s, title, xlabel, ylabel):
    happy_x, happy_y = sort_pair_ascending_xs(h[0], h[1])
    angry_x, angry_y = sort_pair_ascending_xs(a[0], a[1])
    neutral_x, neutral_y = sort_pair_ascending_xs(n[0], n[1])
    sad_x, sad_y = sort_pair_ascending_xs(s[0], s[1])

    fig, ax = plt.subplots()
    scale = 30
    
    ax.set_title(title, y=-0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.scatter(happy_x, happy_y, c='#8dff33', s=scale, label='Happy', edgecolors='none')
    ax.scatter(angry_x, angry_y, c='#ff5733', s=scale, label='Angry', edgecolors='none')
    ax.scatter(neutral_x, neutral_y, c='#ffd133', s=scale, label='Neutral', edgecolors='none')
    ax.scatter(sad_x, sad_y, c='#3374ff', s=scale, label='Sad', edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    
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
            ax.text(x, y, str(float(c))[:5], ha='center',
                    va='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax
