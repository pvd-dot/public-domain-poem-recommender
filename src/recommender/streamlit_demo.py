"""Module for running the recommender with a GUI using the streamlit library."""
import streamlit as st
import vector_searcher as vs
import chatgpt
import recommender


def main():
    # remove excess whitespace at top of page
    st.markdown(
        """
            <style>
                .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """,
        unsafe_allow_html=True,
    )

    if "recs" not in st.session_state:
        chat = chatgpt.ChatGPT()
        vectorsearcher = vs.VectorSearch()
        st.session_state.recs = recommender.Recommender(vectorsearcher, chat)

    st.title("Public Domain Poetry Recommender")

    user_input = st.text_input(
        "Poem Request", placeholder="Can you recommend a short poem about fall?"
    )

    if user_input:
        explanation, poem_text = st.session_state.recs.ask(user_input)
        st.write(explanation)
        st.text(poem_text)


if __name__ == "__main__":
    main()
