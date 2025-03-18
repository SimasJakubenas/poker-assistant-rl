import streamlit as st


def page_summary():

    st.write("## Reinforcement Learnng Poker")
    
    st.image("src/images/game_view.png")
    
    st.write("### Quick Project Summary")

    st.info(
        f"This project aims to train reinforcement model agents to play 6 handed  TexasHoldem poker\n\n"
        f"**Project Terms & Jargon**\n"
        f"* **Hero** a player of focus.\n"
        f"* **BB** a andatory bet at the start of the hand. Also An initial minumum bet.\n"
        f"* **SB** a andatory bet at the start of the hand. half of BB.\n"
        f"* A **Hand** a complete round of play.\n"
        f"* **Hole cards** two cards that are dealt to the player only visible to themselves unless.\n"
        f"the hand goes to showdown.\n"
        f"* **Showdown** a conclusive stage of the hand when all remaining players show their hands \n"
        f"and there is more than 1 player left.\n"
        f"* **Community cards** cards that are dealt on the table and are available to tall players.\n"
        f"* **PREFLOP** an initial state of the hand before community cards are dealt. it is initiated \n "
        f"and concluded by a round of betting.\n"
        f"* **FLOP** Next stage where 3 community cards are shown prior to a round of betting.\n "
        f"* **TURN** Next stage where 1 community card is shown prior to a round of betting.\n "
        f"* **RIVER** Final stage where 1 community card is shown prior to a round of betting.\n\n"
        f"* **Pot** total amount playes have bet.\n\n"
        f"* **Rake** amount taked from the pot by the poker website.\n\n"

        f"**Project Dataset**\n"
        f"* The dataset for this project is a personal hand history of 12000+ hands from a \n "
        f"poplular poker website provided by the developer "
        )

    st.success(
        f"The project has 2 requirements:\n"
        f"* 1 - Gathering my own personal game history data from a popular poker website, "
        f"plotting the data on a graph and determining my win rate. "
        f"churned customer.\n"
        f"* 2 - Training an agent that can beat the game with higher win rate then my own."
        )
      