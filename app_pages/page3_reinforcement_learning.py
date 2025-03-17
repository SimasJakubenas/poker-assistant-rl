import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

COLOR1 = '#26b6a9'
COLOR2 = 'white'
BACKGROUND_COLOR = '#262730'
ACCENT_COLOR1 = '#3097CD'
ACCENT_COLOR2 = '#EF3535'

def reinforcement_learning():

    st.write("### Reinforcement Learning")
    st.info(
        f"* We will attempt to train dqn agents with reinforcement learning and will try and reach a conclusive "
        f"result. \n"
        f"* Tabs represent test results at different stages of the project."
    )

    st.write("---")
    
    # Specify the parent folder path
    parent_folder = "outputs/models/rl"

    # Get all folder names in the parent folder
    folder_names = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]

    train_result_description = [
        {
        "intro": # 0
            """Initial training result. Few things sprung to attention. Fistly - all agents had the same results.
            Secondly and most importantly - all agents had possitive results. This indicates that there is a mistake in 
            calculating rewards""",
        "resolution":
            """At this stage I have not introduced rake and the player reward is equal to balance so the sum of
            all player balances should be 0 meaning that some players should have negative balance. Uppon investigation I've
            noticed a couple mistakes in calculating reward/balance. Those were fixed for other training iterations"""
        },
        {
        "intro": # 1
            """A good split in variance for the agent balances is what one would expect to see. This looks promissing. """,
        "resolution":
            """We've done evaluation for the best performing RL agaent against 5 pre-trained agains"""
        },
        {
        "intro": # 2
            """I've implemented memory replay and made sure it remembers and replays whole hands instead of singe actions'""",
        "resolution":
            """This seemed to have screwed the dabalances in an unatural way. trying different batch variations 
            had no effect on the outcome."""
        }
    ]
    
    tabs = st.tabs(folder_names)
    tab_counter = 0
    
    for i, tab in enumerate(tabs):
        with tab:
            st.write(train_result_description[i]['intro'])
            plot_agents_performance(folder_names[i])
            st.write(train_result_description[i]['resolution'])
            tab_counter += 1
            if i == 1:
                folder_path = f"outputs/datasets/dataframe/v0{i+1}"
                if os.path.exists(folder_path): 
                    data = pd.read_csv(f"{folder_path}/eval.csv")
                    data['Total'] = data['3'].cumsum().round(2)
                    win_rate = round(float(data.iloc[-1]['Total'] / 10), 1)
                    plot_win_rate(data)
                    st.write(
                        f"This shows {win_rate} bb/100 win rate. Anything over 10 bb/10 is exeptional ir extremely hard to achieve. "
                        f"the result indicates that there's something inherently wrong with our model."
                        )


def plot_win_rate(df: pd.DataFrame, ) -> None:
    
    plot_coloring(plt)
    # Copied from sl notebook nr 2
    x = np.arange(len(df))
    y = df['Total'].values
    coefficients = np.polyfit(x, y, 1)
    trendline = np.polyval(coefficients, x)

    # Adjust the trendline to start at 0
    trendline = trendline - trendline[0]

    plt.figure(figsize=(12, 5), facecolor=BACKGROUND_COLOR)
    sns.lineplot(data=df, x=df.index, y='Total', label='Total Win/Loss')
    plt.plot(df.index, trendline, label='Trend Line', color='red')
    plt.ylabel("Count", rotation=0, labelpad=30)
    plt.xlabel("Hands", labelpad=30)
    plt.title(f"Win Rate", fontsize=20)
    plt.legend()
    st.pyplot(plt)
                   
# Initialize an empty DataFrame to store balances
balances = pd.DataFrame()

def plot_agents_performance(folder_name):
    # Loop through each player's folder
    for i in range(6):  # Assuming 6 players
        folder_path = f"outputs/models/rl/{folder_name}/player_{i}"
        if os.path.exists(folder_path):
            # Get all files in the folder
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            files.sort()  # Ensure files are in order
            
            # Extract 'total' from each file and add to the DataFrame
            totals = []
            for file in files:
                checkpoint = torch.load(file)
                total = checkpoint.get('total', None)  # Safely get 'total' from the checkpoint
                totals.append(total)
            
            # Add the totals as a column to the DataFrame
            balances[f"Player_{i}"] = pd.Series(totals)
        else:
            print(f"Folder {folder_path} does not exist.")

    # Smoothen the data using a moving average
    window_size = 10  # Adjust the window size for smoothing
    smoothed_balances = balances.rolling(window=window_size, min_periods=1).mean()

    plot_coloring(plt)
    
    # Plot the smoothed balances for each player
    plt.figure(figsize=(12, 6), facecolor=BACKGROUND_COLOR)
    for player_id in smoothed_balances.columns:
        plt.plot(smoothed_balances[player_id], label=f"{player_id}")

    plt.xlabel("Episodes * 1k")
    plt.ylabel("Balance", rotation=0, labelpad=30)
    plt.title("Player Balances Over Episodes", fontsize=20)
    plt.legend(title="Players")
    plt.grid(True)
    st.pyplot(plt)

def plot_coloring(plt: plt.figure) -> None:
    
    plt.rcParams['text.color'] = COLOR2
    plt.rcParams['axes.labelcolor'] = COLOR2
    plt.rcParams['xtick.color'] = COLOR2
    plt.rcParams['ytick.color'] = COLOR2
    plt.rcParams['grid.color'] = COLOR1

    ax = plt.axes()
    plt.setp(ax.spines.values(), color=COLOR1)