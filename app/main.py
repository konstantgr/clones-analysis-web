import streamlit as st

from multipage import MultiPage
from pages import data_examples, info, clones_results, aggregated_results


if __name__ == "__main__":
    app = MultiPage()

    st.title("Large-scale Python clones analysis ")

    app.add_page("Informational page", info.app)
    app.add_page("Clones results", clones_results.app)
    app.add_page("Aggregated results", aggregated_results.app)

    app.run()
