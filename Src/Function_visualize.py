import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math


def visualize_box(_df, target, nb_col=2):
    """
    Visualizes the distribution of features in a DataFrame using box plots or count plots.

    Parameters:
    _df (DataFrame): The input DataFrame.
    target (str): The target variable to be used for grouping or hue.
    nb_col (int): The number of columns in the subplot grid. Default is 2.

    Returns:
    None
    """
    columns = _df.columns
    nb_row = math.ceil(len(columns)/nb_col)

    fig, axes = plt.subplots(nb_row, nb_col, figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    for i, col_name in enumerate(columns):
        row, col = i // nb_col, i % nb_col
        ax = axes[row, col]
        ax.set_title(f'{col_name}')
        if _df[col_name].dtype == 'object':
            sns.countplot(data=_df, x=col_name, hue=target, ax=ax)
        else:
            sns.boxplot(data=_df, x=target, y=col_name, ax=ax)

    plt.show()


# Define a function to visualize data using violin plots
def visualize_violon(_df, target, nb_col=2):
    """
    Visualizes the distribution of features in a DataFrame using violin plots or count plots.

    Parameters:
    _df (DataFrame): The input DataFrame.
    target (str): The target variable to be used for grouping or hue.
    nb_col (int): The number of columns in the subplot grid. Default is 2.

    Returns:
    None
    """
    columns = _df.columns
    nb_row = math.ceil(len(columns)/nb_col)

    # Create a subplot grid with specified number of rows and columns
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    # Iterate over each column in the DataFrame
    for i, col_name in enumerate(columns):
        row, col = i // nb_col, i % nb_col
        ax = axes[row, col]
        ax.set_title(f'{col_name}')
        
        # If the column is of object type, use countplot
        if _df[col_name].dtype == 'object':
            sns.countplot(data=_df, x=col_name, hue=target, ax=ax)
        else:
            # If the column is numeric, use violinplot
            sns.violinplot(data=_df, x=target, y=col_name, ax=ax)

    # Display the plot
    plt.show()

