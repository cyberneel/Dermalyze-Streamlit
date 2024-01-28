import streamlit as st

def print_pred(pred):
    st.markdown(f'''<u>**Problem:**</u> <br> <li>{pred[0].capitalize()}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**Info:**</u> <br> <li>{pred[1]}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**Symptoms:**</u> <br> <li>{pred[2]}</li>''', unsafe_allow_html=True) 
    st.markdown(f'''<u>**Causes:**</u> <br> <li>{pred[3]}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**More Infor:**</u> <br> <li>{pred[4]}</li>''', unsafe_allow_html=True)