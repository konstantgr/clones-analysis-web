import streamlit as st
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from os import listdir
from os.path import isfile, join

from .results_processing import get_results, pos2code, trim_clone

prohibited_keys = ["file_name", "project_name", "initial_tree_length"]


def get_files(path):
    return [path + '/' + f for f in listdir(path) if isfile(join(path, f)) if '.json' in f]


def get_clones_data(filename, data):
    with open(filename, 'r') as f:
        raw_json = json.load(f)

    clones_count = []
    for key in raw_json.keys():
        if key not in prohibited_keys:
            clones_count.append(json.loads(raw_json[key])['totalClonesCount'])

    x = [int(i) for i in raw_json.keys() if i not in prohibited_keys]

    norm = int(raw_json['initial_tree_length'])
    for i, x_i in enumerate(x):
        data.append((x_i, clones_count[i] / norm))

    return data


def plot_data(fig, ax, data, label, color1='k', color2='r', log=False):
    means = {i: [] for i in range(1, 50, 2)}
    x, y = np.array(data).T

    if not log:
        ax.scatter(x, y, color=color1, s=0.05)
    else:
        ax.scatter(x, np.log(y), color=color1, s=0.05)

    for i in range(len(x)):
        means[x[i]].append(y[i])

    means = {key: np.mean(value) if len(value) else 0 for key, value in means.items()}
    x, y = np.array(list(means.items())).T

    if not log:
        ax.scatter(x, y, color=color2, s=30, label=label)
    else:
        ax.scatter(x, np.log(y), color=color2, s=30, label=label)

    return y


def app():
    # Get directories of results and code sources
    default_value = '/Users/konstantingrotov/Documents/Programming/projects/clone-analysis-web/data/scripts'
    notebook_results_directory = st.text_input('Absolute path to directory with notebooks results', value=default_value)

    default_value = '/Users/konstantingrotov/Documents/Programming/projects/clone-analysis-web/data/notebooks'
    scripts_results_directory = st.text_input('Absolute path to directory with scripts results', value=default_value)

    notebooks_files = get_files(notebook_results_directory)
    scripts_files = get_files(scripts_results_directory)
    st.write(f"Totally {len(notebooks_files)} notebooks and {len(scripts_files)} scripts")

    data_scripts = []
    data_notebooks = []

    for name in tqdm(scripts_files[30:]):
        data_scripts = get_clones_data(name, data_scripts)

    for name in tqdm(notebooks_files[:]):
        data_notebooks = get_clones_data(name, data_notebooks)

    fig, ax = plt.subplots()

    s_means = plot_data(fig, ax, data_scripts, "scripts", log=False)
    n_means = plot_data(fig, ax, data_notebooks, "notebooks", 'g', 'b', log=False)

    ax.set_ylim(-0.01, 0.4)
    ax.set_xlim(0, 30)

    ax.set_xlabel("Min clone length")
    ax.set_ylabel("Total clones count / PSI Tree size")

    ax.legend()

    st.pyplot(fig)

    fig, ax = plt.subplots()

    ax.scatter(range(1, 50, 2), n_means / s_means, color='r')

    ax.set_xlabel('minCloneLength')
    ax.set_ylabel("ntb_value / scr_value")
    st.pyplot(fig)
