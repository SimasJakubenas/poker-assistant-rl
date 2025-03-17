import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


COLOR1 = '#26b6a9'
COLOR2 = 'white'
BACKGROUND_COLOR = '#262730'
ACCENT_COLOR1 = '#3097CD'
ACCENT_COLOR2 = '#EF3535'

sns.set_theme(style="whitegrid", rc={"axes.facecolor": BACKGROUND_COLOR}) 

def supervised_learning():
    
    st.write("## Supervised Learning")
    st.write("### Data Collection")
    st.write(
        f"Data was collected from developers personal poker hand history. While it came "
        f"in what would be considered a standart format for such data, it wasn't very convenient "
        f"to deal with and required great attention to parsing it. It doen't list total values "
        f"gained/lossed of **hero** player which we were looking and it had to be calculated precisely."
        f"Parsed data was added to dataframe which you can see below."
        )
    
    if st.checkbox("Inspect Hand Data"):
        st.image("src/images/hand_example.png")
    
    
    df = pd.read_csv('outputs\datasets\dataframe\hero_winrate.csv')

    if st.checkbox("Parsed Data Dataframe"):
        st.write(
            f"* The dataset has {df.shape[0]} rows. "
            f"You can see the first 5 rows displayed below:"
            )
        
        st.dataframe(df.head(5))

    st.write("### Hand History Study")
    st.write("We are plotting heros win rate result.")
    
    plot_win_rate(df)
    
    win_rate = round(float(df.iloc[-1]['Total'] * 10 /100), 1)
    st.write(
        f"Hero's win rate is **{win_rate} bb/100**. this expression (bb/100) is used commonly amongst "
        f"poker players to measure someone's winrate. This expression normalises the value independenly of "
        f"stakes or number of hands played per hour. {win_rate} bb/100 is a good win rate"
        )
    st.write(
        f"This study answers **requirement 1** (hero has a possitive win rate) and we will be using  "
        f"this data to pre-train our agents"
        )
    
    st.write("### Agent training with SL")
    st.write(
        f"We parsed the hand hsitory again. This time with different custom function as we needed to "
        f"extract more data (betting rounds, community cards, other player win/loss etc...)"
        f"Parsed data then was matched with game state for continuity. 16 features were selected/created "
        f"ten normalised and passed as state vectors to the input layer of neural network. These features are: "
        f"**stack size**, **current bet**, **comunity cards**, **pot**, **player possition**, **stack to pot ratio** "
        f"**bet to pot ratio** **active players**, **active status**, **fold status**, **all in status**, "
        f"and 4 boolean values representing **betting rounds**"
        )
    
    if st.checkbox("Show Model"):
        st.image("src/images/model.png")
    
    st.write(
        f"Model shows loss as mse 5,3 and 5,6 for train and test splits respectively. It does now appear that the "
        f"model is training at all, but I'm out of time and ideas so I'm plowing ahead."
        )
    
    st.image("src/images/sl_training_results.png")

    

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

def plot_coloring(plt: plt.figure) -> None:
    
    plt.rcParams['text.color'] = COLOR2
    plt.rcParams['axes.labelcolor'] = COLOR2
    plt.rcParams['xtick.color'] = COLOR2
    plt.rcParams['ytick.color'] = COLOR2
    plt.rcParams['grid.color'] = COLOR1

    ax = plt.axes()
    plt.setp(ax.spines.values(), color=COLOR1)
