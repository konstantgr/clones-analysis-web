import streamlit as st
import os
from .results_processing import get_results, pos2code, trim_clone


def get_code(path):
    with open(path, 'r') as f:
        return f.read()


def app():
    # Get directories of results and code sources
    # default_value = '/Users/konstantingrotov/Documents/Programming/tools/clones-analysis/validation_anon/not_anon'
    default_value = os.path.abspath("data/results_data/results")
    results_directory = st.text_input('Absolute path to directory with results', value=default_value)

    # default_value = '/Users/konstantingrotov/Documents/Programming/datasets/Lupa-duplicates/data/test_on_anon'
    default_value = os.path.abspath("data/results_data/sources")
    sources_directory = st.text_input('Absolute path to directory with source codes', value=default_value)

    # Get all results
    results_dict = {
        r.name: r for r in get_results(results_directory)
    }

    # Select particular project
    option = st.selectbox('Select project', [r for r in results_dict.keys()])
    res = results_dict[option]

    # Get clones of this project
    try:
        source_code = get_code(os.path.join(
            sources_directory, f"{res.owner}#{res.repository}/{res.name}"
        ))
    except Exception as e:
        source_code = None

    min_length = st.select_slider('Select minimum length of clones', options=res.clones_lengths)
    positions = res.get_clones_stats(min_length)

    if source_code:
        with st.expander(f"code"):
            st.code(source_code)

        for i, group in enumerate(positions):
            with st.expander(f"Group {i}"):
                for poses in group:
                    # trim_clone(poses, source_code)
                    clone, start_line, end_line = pos2code(poses, source_code)
                    st.caption(f'{start_line} => {end_line}')
                    st.caption(f'Length of sequence is {len(poses)}')

                    st.code(clone)
