import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt, mpld3
from matplotlib.figure import Figure
from mpld3 import fig_to_html
import openai
import numpy as np
import random
import requests
import pickle
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from matplotlib.colors import to_hex
import os
from os import listdir
from dotenv import load_dotenv

load_dotenv()

token = "pk-zIqZkDvGIHPsmpxWuBnkFuvvWglIiZJheWkTzljQgmxUBvBt"
openai.api_key = token
openai.api_base = "https://api.pawan.krd/v1"

### Customize UI
st.set_page_config(layout="wide", page_title="Chat2PyViz", page_icon=":python:")

st.markdown(
    """ <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """,
    unsafe_allow_html=True,
)

st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {0}rem;
        padding-right: {5}rem;
        padding-left: {3}rem;
        padding-bottom: {0}rem;
    }} </style> """,
    unsafe_allow_html=True,
)


### Session ID
def get_session_id():
    session_id = get_report_ctx().session_id
    session_id = session_id.replace("-", "_")
    session_id = "_id_" + session_id
    return session_id


st.session_state.id = str(get_session_id())


### Init Figure
def init_widgets():
    if "xaxis" in st.session_state:
        del st.session_state.xaxis
    if "yaxis" in st.session_state:
        del st.session_state.yaxis
    if "title" in st.session_state:
        del st.session_state.title
    if "xlim_start" in st.session_state:
        del st.session_state.xlim_start
    if "xlim_end" in st.session_state:
        del st.session_state.xlim_end
    if "ylim_start" in st.session_state:
        del st.session_state.ylim_start
    if "ylim_end" in st.session_state:
        del st.session_state.ylim_end
    return True


### Layout Sidebar
# API Configuration
st.sidebar.markdown("### API Configuration ###")

options = ["OpenAI API", "HuggingFace API", "Free OpenAI API"]
selected_option = st.sidebar.radio("Select an option:", options)

if "OpenAI API" in selected_option:
    api_token = st.sidebar.text_input("API Token")

# elif "HuggingFace API" in selected_option:
#     api_token = st.sidebar.text_input("API Token")

# elif "Free OpenAI API" in selected_option:
#     api_token = st.sidebar.text_input("API Token")
#     api_base = st.sidebar.text_input("API Base URL")

#     # personal token: "pk-zIqZkDvGIHPsmpxWuBnkFuvvWglIiZJheWkTzljQgmxUBvBt"
#     openai.api_key = api_token
#     openai.api_base = "https://api.pawan.krd/v1"

st.sidebar.markdown("### Datasets ###")

# Data Selection
datafiles = ["No Dataset"] + [file for file in listdir(".") if file.endswith(".csv")]
option = st.sidebar.selectbox(
    "Which dataset do you want to use?",
    datafiles,
    key="dataset",
    on_change=init_widgets,
)

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Or upload a file:", type=["csv"], key="fileupload", on_change=init_widgets
)
if uploaded_file != None:
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        df.to_csv(uploaded_file.name)
        datafiles.append(uploaded_file.name)
        option = uploaded_file.name
    except:
        st.sidebar.write("Could not import")

if option in datafiles:
    ind = datafiles.index(option)
    if option != "No Dataset":
        df = pd.read_csv(datafiles[ind], index_col=0)
        # print(df.head(6))
        data = pd.DataFrame(df.dtypes)
        data = data.astype(str)
        data.columns = ["Type"]
        data = data.Type.replace(
            {"object": "String", "float64": "Float", "int64": "Int"}
        )
        st.sidebar.table(data)
        # print(data)
        st.session_state.csv = datafiles[ind]
        st.session_state.load_data = data.to_string()
    else:
        st.sidebar.write("No Dataset selected")
        st.session_state.csv = ""
        st.session_state.load_data = ""


# Execute the generated code
def my_exec(script):
    """Execute a script protected"""
    scriptlist = script.split("\n")
    scriptlist = [scr for scr in scriptlist if not scr.endswith(".show()")]
    for scr in scriptlist[:]:
        try:
            exec(scr)
            print(scr)
        except:
            pass
    return None


### Layout Main ###
col1, col2 = st.columns([8, 2])


def set_widgets():
    with col2:
        pass


### Query Code from getGPT3
def getGPT3():
    on_change = init_widgets()
    if st.session_state.comand_input == "":
        expr = ""
    else:
        messages = [
            {
                "role": "system",
                "content": f"""
                You are data visualization bot, which transforms the user's natural language instruction into the data visualization Python code.
                the code should include the import of csv data file, which is {st.session_state.csv}, the creation of a dataframe variable df, and the creation of a figure variable fig.
                The dataframe variable df includes these data columns and data types in them: {st.session_state.load_data}.
                Remember you only need to return Python code. 
            """,
            },
            {
                "role": "user",
                "content": st.session_state.comand_input,
            },
        ]
        # print("!!!" + messages + "!!!")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", max_tokens=1024, messages=messages
        )
        expr = response["choices"][0]["message"]["content"]
        print(expr)
        st.session_state["comand_output"] = expr

    return True


### Create Figure ###


# @st.cache
def create_figure():
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (10, 5)

    if "comand_output" in st.session_state:
        my_exec(st.session_state.comand_output)

    fig = plt.gcf()

    return fig


with col1:
    # st.write(st.session_state)
    st.header("Create Charts with Commands in Natural Language")
    demo_video = st.expander(label="Tutorial Video")
    with demo_video:
        # video_file = open('NLP2Chart.mp4', 'rb')
        # video_bytes = video_file.read()
        st.video(data="https://youtu.be/UiCSczhslAs")
    st.text_input(
        "Advise the system",
        key="comand_input",
        on_change=getGPT3,
        help="Examples: \n Plot a sinus function from -4 pi to 4 pi; \n Make an array of 400 random numbers and plot a horizontal histogram; \n plot sum of total_cases grouped by location as bar chart (COVID19 Data)",
    )
    if "comand_output" in st.session_state:
        st.code(st.session_state.comand_output, language="python")
        fig = create_figure()
        st.pyplot(fig=fig)


set_widgets()

### Export Figures

# st.sidebar.markdown("### Export ###")

# with open("fig" + st.session_state.id + ".pickle", "rb") as f:
#     fig = pickle.load(f)
# fig.savefig("figure_export.png", dpi=fig.dpi)
# mpld3.save_html(fig, "figure_export.html")
# with open("figure_export.png", "rb") as f:
#     st.sidebar.download_button("Download PNG", f, file_name="figure_export.png")
# with open("figure_export.html", "rb") as f:
#     st.sidebar.download_button("Download HTML", f, file_name="figure_export.html")
