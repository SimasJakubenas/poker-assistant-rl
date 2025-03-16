import streamlit as st

from app_pages.multipage import MultiPage
from app_pages.page1_summary import page_summary

app = MultiPage(app_name= "Poker Assistant")

app.app_page("Project Summary", 'house', page_summary)


app.run()