# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:42:35 2023

@author: Ryan Hannan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lineplot(df, headers):
    """
    A very simple function to create lineplots.

    Parameters
    ----------
    df : A dataframe object.
    headers : List of columns passed to the function to plot our graphs

    Returns
    -------
    None.
    """
    # Configuring our plot
    plt.figure(figsize=(15, 10), dpi=100)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    """Loop through the header arguments for the number of plots we want to
    generate"""
    for head in headers:
        plt.plot(df['fDate'], df[head], label=head)

    plt.title('Fuel Prices', fontdict={'family': 'serif',
                                       'color':  'grey',
                                       'weight': 'bold',
                                       'size': 25,
                                       })
    # Labelling our data
    plt.grid(True)
    plt.xlabel('Timeline')
    plt.ylabel('Cents / litre')
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df['fDate'].min(), df["fDate"].max())
    plt.legend()
    plt.savefig('lineplot.png')
    plt.show()

    return


def pieplot(df, headers):
    """
    Takes some parameters to generate a pie plot.

    Parameters
    ----------
    df : A dataframe object.
    headers : List of arguments we are interested in plotting.

    Returns
    -------
    None.

    """
    # Plot configuration
    plt.figure(figsize=(15, 10), dpi=100)
    plt.pie(df, labels=headers, autopct='%.1f%%', explode=(0, 0.07),
            colors=['dodgerblue', 'red'],
            wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
            textprops={'size': 'x-large'})
    plt.title('Cereal Type Proportions', fontdict={'family': 'serif',
                                                   'color': 'grey',
                                                   'weight': 'bold',
                                                   'size': 25})
    plt.savefig('pieplot.png')
    plt.show()

    return


def histplot(df):
    """
    Function generates a hist plot given a dataframe

    Parameters
    ----------
    df : Only a dataframe object passed to the function

    Returns
    -------
    None.

    """
    # Configuring our plot
    plt.figure(figsize=(15, 10), dpi=70)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    plt.hist(df['Fatalities'], bins=70, edgecolor='black', log=True)
    plt.title('Aircraft Crashes & Fatalities', fontdict={'family': 'serif',
                                                         'color':  'grey',
                                                         'weight': 'bold',
                                                         'size': 25})
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Fatalities')
    plt.ylabel('Crashes')
    plt.legend()
    plt.savefig('histoplot.png')
    plt.show()

    return


# Load fuel prices data into a dataframe
fuel_prices = pd.read_csv('int_fuel.csv')

# Taking a look at our Data
print(fuel_prices.head())
print(fuel_prices.describe())

# Selecting the columns of interest
fuel_df = fuel_prices.iloc[:, 1:8]

# Formatting the DateTime data to suit our plots
fuel_df['Year'] = pd.DatetimeIndex(fuel_df['Date']).year
fuel_df['fDate'] = pd.to_datetime(fuel_df['Date']).dt.strftime('%Y %m')
print(fuel_df.columns)
# Rename columns for easier access
fuel_df.rename(columns={'UK/Royaume-Uni': 'UK', 'Germany/Allemagne': 'Germany',
                        'Japan/Japon': 'Japan',
                        "USA/États-Unis d'Amérique": 'USA'}, inplace=True)

# Selecting our data for year 2019
year_df = fuel_df[fuel_df['Year'] == 2019]

# Calling lineplot function to plot a graph for the year 2019
lineplot(year_df[::3], ['UK', 'Canada', 'USA', 'Japan', 'France'])

crash = pd.read_csv('crash.csv')
# Create a year column from the Date column to use in our graphs
crash['Year'] = pd.DatetimeIndex(crash['Date']).year
crash['Survivors'] = crash['Aboard'] - crash['Fatalities']

# Calling the histplot function
histplot(crash)

cereal = pd.read_csv('cereal.csv')
print(cereal.head())
# Calculate C & H percentage
mask = cereal['type'] == 'C'
c_count = len(cereal[mask])
print(cereal['type'].isnull().values.any())
h_count = len(cereal[cereal['type'] == 'H'])
print(h_count)
perc = np.array([(c_count/len(cereal)) * 100, (h_count/len(cereal) * 100)])
print(perc)

# Calling pie plot
pieplot(perc, ['C', 'H'])
