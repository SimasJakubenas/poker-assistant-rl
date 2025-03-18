# Reinforcement Learning poker assistant 

This is my Final Project for the studies with [codeacademy.lt](https://codeacademy.lt/) .
With this project I will be attempting to create Deep Q-Learning model that trains agents to play a 6 handed no-limit Texas Holdem poker.
Link to dashboard - [poker rl](https://poker-rl.streamlit.app/)
---

## Contents

* [Dataset Content](#dataset-content) 🗃️

* [Business Requirements](#business-requirements) 📋

* [Hypothesis and Validation Approach](#hypothesis-and-validation-approach) 💡

* [Rationale](#the-rationale) ✍

* [Dashboard Design](#dashboard-design) 📐

* [Unfixed Bugs](#unfixed-bugs) 🛠️

* [Deployment](#deployment) 🖥️

* [Data Analysis and ML Libraries](#data-analysis-and-ml-libraries) 📚

* [Credits and Acknowledgments](#credits-and-acknowledgments) 💐

## Dataset Content

The dataset for this project is developers personal hand history of 12000+ hands from a poplular poker website

## Requirements

These are the requirements for the project:
1 - Gathering my own personal game history data from a popular poker website, plotting the data on a graph and determining my win rate.

2 - Training an agent that can beat the game with higher win rate then my own.

## Hypothesis and Validation Approach

1. I myself have a positive win rate.
* Data Visualisation

2. Reinforcement Learning agent can beat my win rate.
* Deep Q-Learning Neural Network

3. Reinforcement Learning agent can beat the game after the introduction of rake (rake is a percentage a poker website takes from the money in the middle once the hand is over).
* Deep Q-Learning Neural Network

## The Rationale

**Requirement 1:** Data Visualisation
* We will gather my own hand history from a popular poker website.
* We will visualise the parsed data.

**Requirement 2:** Deep Q-Learning Neural Network
* We will create the environment to train our agents in.
* We will pre-train agents with the data we collected for 'Requirement 2'.
* We will differentiate the reward system to try and immitate different play styles.
* We will continue training agents for a significant amount of time.
* We will pick the best performing agent and eveluate the results.

## Dashboard Design

### Page 1 
### Project Summary
* Project Terms & Jargon
* Describe Project Dataset
* State Requirements

The Project Summary page outlines the project's terminology and jargon, provides a description of the dataset, and details the requirements. Users can gain a comprehensive overview of the project here, along with access to the readme file and a link to the dataset on Kaggle.

The variable for the dataset are explained in detail:

![dashboard one](src/images/readme/page_1.png)

### Page 2
### Supervised Learning Study
* This page fulfills Business Requirement 1.

    * data structure
    
    ![dashboard two](src/hand_example.png)

    * shows hero's win rate
    
    ![dashboard two](src/images/readme/page_2.png)

    * aend SL trainign results

### Page 3
### Heart Disease Prediction
* This page shows steps taken in RL process
    
    ![dashboard three](src/images/readme/page_3.png)

### Page 4
### Hypothesis and validation
* Shows variance calculator
    
    ![dashboard four](src/images/readme/page_3.png)

## Unfixed Bugs

### BUG 1
* Plyer able to raise when every other player is all in (doen't effect the game but will have inpack when rake is introduced)

### BUG 2
* When player goes all-in and and the raise is less than current min raise others should only be ble to call the raise. At the moment players can re-raise also

### BUG 2
* If player puts in more chips than anybody else they should get those chips back prior to showdown. Now they get it back as side pot at the end of hand. Will be a problem when rake is introduced

## Usage

### Running the UI

The pre-trained model is included in the repository however you can pre-train the model by using the provided notebook in supervised_learning/

to run dashboard localy type streamlit run app.py in the terminal

```bash
python main.py --mode ui --n_players 6 --human_player 0 --random_agents False
```

### Training an Agent

```bash
python main.py --mode train --episodes 100000 --save_path v0x
```

### Evaluating a Trained Agent

```bash
python main.py --mode evaluate --save_path v0x --select_agent 0
```

## Command-line Arguments

- `--mode`: Run mode (ui, train, or evaluate)
- `--n_players`: Number of players (2-6)
- `--human_player`: ID of human player (0-5), default is None (all AI)
- `--small_blind`: Small blind amount
- `--big_blind`: Big blind amount
- `--initial_stack`: Initial stack size
- `--episodes`: Number of episodes to train (training mode)
- `--batch_size`: Batch size for training (training mode)
- `--target_update`: Episodes between target network updates (training mode)
- `--save_path`: Path to save/load the trained model
- `--select_agent`: Selects which agent to evaluate on


## Data Analysis and ML Libraries

* [Pandas](https://pandas.pydata.org/) for data analysis, exploration, manipulation and visualization e.g.e create dataframes throughout the Jupyter Notebooks
* [NumPy](https://numpy.org/) was used to process arrays and data 
* [Matplotlib](https://matplotlib.org/) for graphs and plots
* [Seaborn](https://seaborn.pydata.org/) to visualize the data in the Streamlit app with graphs and plots
* [torch](https://pytorch.org/) to utilise all functionality in regards to neural networks

## Credits and Acknowledgments 

### Content

- Most of the game was generated by ClaueAI, however it came with countless bugs that had to be resolved

### Media

* Media are screenshots from my notebooks and dashboard.

### Acknowledgements

### Acknowledgements
* God for giving me strenght every day.
* My tutors Rokas Slabosevicius ir Justas Kvederis for their endless wisdom, patience and continuos support.
* My fellow students at codeacademy.lt for making it an enjoyable ride.
* Sorry to friends and family for my absence during these few months.
