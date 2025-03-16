import streamlit as st
from streamlit_option_menu import option_menu


class MultiPage: 

    def __init__(self, app_name: str) -> None:
        self.pages = []
        self.app_name = app_name
        
        if app_name not in st.session_state:
            st.session_state['home'] = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="suit-spade-fill")
    
    
    def app_page(self, title: str, icon: str, func) -> None: 
        self.pages.append({"title": title, "icon": icon, "function": func})


    def run(self) -> None:
        with st.sidebar:
            selected = option_menu(
                menu_title=None,
                options=[x['title'] for x in self.pages], 
                icons=[x['icon'] for x in self.pages],
                menu_icon="cast",
                default_index=0
                )
        
        for page in self.pages:
            if page['title'] == selected:
                page['function']()


