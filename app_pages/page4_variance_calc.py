import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


COLOR1 = '#26b6a9'
COLOR2 = 'white'
BACKGROUND_COLOR = '#262730'
ACCENT_COLOR1 = '#3097CD'
ACCENT_COLOR2 = '#EF3535'

def variance_calculator():
    st.write("### Poker Variance Calculator")
    st.info(
        f"This tool simulates poker variance based on your win rate (bb/100) and the number of hands played. "
        f"It generates multiple variance spreads to help you visualize potential outcomes."
    )

    st.write("---")

    # User inputs
    winrate = st.number_input("Enter your win rate (bb/100):", value=5.0, step=0.1)
    hands = st.number_input("Enter the number of hands:", value=10000, step=1000)
    num_simulations = st.slider("Number of variance simulations:", min_value=1, max_value=10, value=5)

    st.write("---")

    # Variance simulation
    if hands > 0 and num_simulations > 0:
        bb_per_hand = winrate / 100  # Convert bb/100 to bb per hand
        std_dev = 80 / np.sqrt(100)  # Approximate standard deviation for poker in bb/hand

        # Generate variance spreads
        simulations = []
        final_values = []  # To store the final cumulative values
        for _ in range(num_simulations):
            results = np.random.normal(loc=bb_per_hand, scale=std_dev, size=int(hands))
            cumulative_results = np.cumsum(results)
            simulations.append(cumulative_results)
            final_values.append(round(cumulative_results[-1], 2))  # Store the last value

        plot_coloring(plt)

        # Plot the variance spreads
        plt.figure(figsize=(12, 6), facecolor=BACKGROUND_COLOR)
        for sim in simulations:
            plt.plot(sim, alpha=0.7)
        plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Break-even")
        plt.title(f"Variance Simulation for {hands} Hands at {winrate} bb/100", fontsize=20)
        plt.xlabel("Number of Hands", labelpad=30)
        plt.ylabel("Cumulative bb", rotation=0, labelpad=30)
        plt.legend(["Variance Spreads", "Break-even"])
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Display the final cumulative results as a table
        st.write("### Final Cumulative Results for Each Simulation")
        final_results_df = {"Simulation": [f"Simulation {i+1}" for i in range(num_simulations)],
                            "Final Value (Cumulative bb)": final_values}
        st.dataframe(final_results_df)

    else:
        st.warning("Please enter a valid number of hands and simulations.")


def plot_coloring(plt: plt.figure) -> None:
    plt.rcParams['text.color'] = COLOR2
    plt.rcParams['axes.labelcolor'] = COLOR2
    plt.rcParams['xtick.color'] = COLOR2
    plt.rcParams['ytick.color'] = COLOR2
    plt.rcParams['grid.color'] = COLOR1

    ax = plt.axes()
    plt.setp(ax.spines.values(), color=COLOR1)