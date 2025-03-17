import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.page1_summary import page_summary
from app_pages.page2_supervised_learning import supervised_learning
from app_pages.page3_reinforcement_learning import reinforcement_learning
from app_pages.page4_variance_calc import variance_calculator

app = MultiPage(app_name= "Poker Assistant")

app.app_page("Project Summary", 'house', page_summary)
app.app_page("Supervised Learning", 'graph-up-arrow', supervised_learning)
app.app_page("Reinforcement Learning", 'arrow-repeat', reinforcement_learning)
app.app_page("Variance Calculator", 'book-half', variance_calculator)

app.run()