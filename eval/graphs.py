import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def one_emotion_four_reverb_factors(emotion_x, acc):
    x_measurement = [2, 4, 6, 8]
   
    emotion_x, acc = sort_pair_ascending_xs(list(emotion_x), list(acc))
    two_acc = []
    four_acc = []
    six_acc = []
    eight_acc = []
    
    for index in range(0, len(emotion_x)):
        if emotion_x[index] == 2:
            two_acc.append(acc[index])
        elif emotion_x[index] == 4:
            four_acc.append(acc[index])
        elif emotion_x[index] == 6:
            six_acc.append(acc[index])
        elif emotion_x[index] == 8:
            eight_acc.append(acc[index])
    
    return two_acc, four_acc, six_acc, eight_acc


def draw_scatter_reverb_factors(h, a, n, s, title, xlabel, ylabel):
    happy_x, happy_y = sort_pair_ascending_xs(h[0], h[1])
    angry_x, angry_y = sort_pair_ascending_xs(a[0], a[1])
    neutral_x, neutral_y = sort_pair_ascending_xs(n[0], n[1])
    sad_x, sad_y = sort_pair_ascending_xs(s[0], s[1])
    
    for element in range(0, len(happy_x)): elements += 0.01
    for element in range(0, len(angry_x)): elements += 0.01
    for element in range(0, len(neutral_x)): elements += 0.01
    for element in range(0, len(sad_x)): elements += 0.01

    fig, ax = plt.subplots()
    scale = 30
    
    ax.set_title(title, y=-0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.scatter(happy_x, happy_y, c='green', s=scale, label='Happy', alpha=0.6, edgecolors='none')
    ax.scatter(angry_x, angry_y, c='red', s=scale, label='Angry', alpha=0.6, edgecolors='none')
    ax.scatter(neutral_x, neutral_y, c='yellow', s=scale, label='Neutral', alpha=0.6, edgecolors='none')
    ax.scatter(sad_x, sad_y, c='blue', s=scale, label='Sad', alpha=0.6, edgecolors='none')
    ax.legend()
    ax.grid(True)

    plt.show()
    

def histogram(h, a, n, s, title, xlabel, ylabel):
    
    x_measurement = [2, 4, 6, 8]
    
    labels = ['2', '4', '6', '8']

    h_x, h_acc = sort_pair_ascending_xs(h[0], h[1])
    two_acc_h, four_acc_h, six_acc_h, eight_acc_h = one_emotion_four_reverb_factors(h_x, h_acc)
    h_means = [np.mean(two_acc_h), np.mean(four_acc_h), np.mean(six_acc_h), np.mean(eight_acc_h)]
    
    a_x, a_acc = sort_pair_ascending_xs(a[0], a[1])
    two_acc_a, four_acc_a, six_acc_a, eight_acc_a = one_emotion_four_reverb_factors(a_x, a_acc)
    a_means = [np.mean(two_acc_a), np.mean(four_acc_a), np.mean(six_acc_a), np.mean(eight_acc_a)]
    
    n_x, n_acc = sort_pair_ascending_xs(n[0], n[1])
    two_acc_n, four_acc_n, six_acc_n, eight_acc_n = one_emotion_four_reverb_factors(n_x, n_acc)
    n_means = [np.mean(two_acc_n), np.mean(four_acc_n), np.mean(six_acc_n), np.mean(eight_acc_n)]
    
    s_x, s_acc = sort_pair_ascending_xs(s[0], s[1])
    two_acc_s, four_acc_s, six_acc_s, eight_acc_s = one_emotion_four_reverb_factors(s_x, s_acc)
    s_means = [np.mean(two_acc_s), np.mean(four_acc_s), np.mean(six_acc_s), np.mean(eight_acc_s)]
    
    x = np.arange(len(x_measurement))  # the label locations
    
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    
    rect1 = ax.bar(x - 2 * width, h_means, width, label='Happy', color='#8dff33')
    rect2 = ax.bar(x - width, a_means, width, label='Angry', color='#ff5733')
    rect3 = ax.bar(x, n_means, width, label='Neutral', color='#ffd133')
    rect4 = ax.bar(x + width, s_means, width, label='Sad', color='#3374ff')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title, y=-0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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
