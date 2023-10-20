### Imports
import streamlit as st
from datetime import datetime
import time

### CSS
st.set_page_config(
    page_title='Example Plage', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
padding_top = 10
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

### UI
""
col1, col2 = st.columns( [1,5] )
col1.image('AderasBlue2.png', width=70)
col1.image('AderasText.png', width=70)
col2.title('Simple Streamlit Example')
# st.markdown('---')

with st.sidebar: 
    mysidebar = st.selectbox('Select Option', ['Option 1', 'Option 2'])
    if mysidebar == 'Option 1':
        show_detail = st.checkbox('Write to sidebar')
        llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
        st.markdown("---")
        tz = st.container()
    if mysidebar == 'Option 2':
        st.markdown("---")
        st.markdown("### There isn't an Option 2:")
        
if mysidebar == 'Option 1':
    with st.expander("**:blue[Streamlit Overvew]**"):
        st.markdown("**:red[Streamlit is an open-source Python library that makes it easy to create custom web apps and dashboards for machine learning and data science projects.]**")
        st.markdown("Streamlit makes it incredibly fast and simple to translate Python data analysis code into a shareable web app or dashboard. It's a great tool for ML engineers and data scientists.")

    st.markdown('---')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Here you would ask questions but there is no code in this app?"):
        start = datetime.now()
        if show_detail:
            tz.write("Start: "+str(start))
        else:
            st.write("Start: "+str(start))
        time.sleep(3) 
        if show_detail:
            tz.write("End: "+str(datetime.now()))
            tz.write("Duration: "+str(datetime.now() - start))
        else:
            st.write("End: "+str(datetime.now()))
            st.write("Duration: "+str(datetime.now() - start))
            st.write("You selected model: "+llm_model)
