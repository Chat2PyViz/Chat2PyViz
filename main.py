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
from pathlib import Path
from streamlit_elements import elements, mui, html, editor, lazy, sync, event, dashboard
from dashboard import Dashboard, Editor, Card, DataGrid, Radar, Pie, Player
from types import SimpleNamespace
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
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
header {visibility: hidden;}
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


# ### Init Figure
# def init_widgets():
#     if "xaxis" in st.session_state:
#         del st.session_state.xaxis
#     if "yaxis" in st.session_state:
#         del st.session_state.yaxis
#     if "title" in st.session_state:
#         del st.session_state.title
#     if "xlim_start" in st.session_state:
#         del st.session_state.xlim_start
#     if "xlim_end" in st.session_state:
#         del st.session_state.xlim_end
#     if "ylim_start" in st.session_state:
#         del st.session_state.ylim_start
#     if "ylim_end" in st.session_state:
#         del st.session_state.ylim_end
#     return True


### Layout Sidebar
# API Configuration
st.sidebar.markdown("### API Configuration ###")

options = ["OpenAI API"]  # , "HuggingFace API", "Free OpenAI API"]
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
)

# Header
st.header("Chat2PyViz")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Or upload a file:",
    type=["csv"],
    key="fileupload",
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
        df = pd.read_csv(datafiles[ind])  # , index_col=0)
        # print(df.head(6))
        st.dataframe(df.head(4))
        data = pd.DataFrame(df.dtypes)
        data = data.astype(str)
        data.columns = ["Type"]
        data = data.Type.replace(
            {"object": "String", "float64": "Float", "int64": "Int"}
        )
        # st.sidebar.table(data)
        # print(data)
        st.session_state.csv = datafiles[ind]
        st.session_state.load_data = data.to_string()
    else:
        # st.sidebar.write("No Dataset selected")
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


### Query Code from getGPT3
def getGPT3():
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
            model="gpt-3.5-turbo", max_tokens=500, messages=messages
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

    fig = None

    if (
        "comand_output" in st.session_state
        and st.session_state.comand_output != "# The code will be shown in this editor."
    ):
        my_exec(st.session_state.comand_output)
        fig = plt.gcf()

    return fig


### Layout Main ###
demo_video = st.expander(label="Tutorial Video")
with demo_video:
    # video_file = open('NLP2Chart.mp4', 'rb')
    # video_bytes = video_file.read()
    st.video(data="https://youtu.be/UiCSczhslAs")
st.text_input(
    "Input the prompt:",
    key="comand_input",
    on_change=getGPT3,
    help="Examples: \n Plot a sinus function from -4 pi to 4 pi; \n Make an array of 400 random numbers and plot a horizontal histogram; \n plot sum of total_cases grouped by location as bar chart (COVID19 Data)",
)
### Columns ###
col1, col2 = st.columns(2)

with col1:
    # st.write(st.session_state)

    if "comand_output" not in st.session_state:
        st.session_state.comand_output = "# The code will be shown in this editor."

    def update_content(value):
        st.session_state.comand_output = value

    # # Next, create a dashboard layout using the 'with' syntax. It takes the layout
    # # as first parameter, plus additional properties you can find in the GitHub links below.
    # def handle_layout_change(updated_layout):
    #     # You can save the layout in a file, or do anything you want with it.
    #     # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
    #     print(updated_layout)

    st.session_state.number = random.randint(0, 10000)
    previous = "# The code will be shown in this editor."

    # st.write(st.session_state)

    if st.session_state.comand_output == previous:
        with elements("code_initial"):
            editor.Monaco(
                height=600,
                defaultValue=st.session_state.comand_output,
                onChange=lazy(update_content),
                language="python",
                theme="vs-dark",
                key="code_editor",
            )
            mui.Button("Run", variant="outlined", onClick=sync())
    else:
        with elements(f"code_editor_{st.session_state.number}"):
            editor.Monaco(
                height=600,
                defaultValue=st.session_state.comand_output,
                onChange=lazy(update_content),
                language="python",
                theme="vs-dark",
                key=f"code_editor_{st.session_state.number}",
            )
            event.Hotkey("ctrl+s", sync(), bindInputs=True, overrideDefault=True)
            mui.Button("Run", onClick=sync())

    # if "w" not in st.session_state:
    #     board = Dashboard()
    #     w = SimpleNamespace(
    #         dashboard=board,
    #         editor=Editor(board, 0, 0, 6, 11, minW=3, minH=3),
    #     )
    #     st.session_state.w = w

    #     w.editor.add_tab("Code", st.session_state.comand_output, "python")
    # else:
    #     w = st.session_state.w

    # if (
    #     "comand_output" in st.session_state
    #     and st.session_state.comand_output != "# The code will be shown in this editor."
    # ):
    #     w.editor.update_content("Code", st.session_state.comand_output)
    # with elements("change"):
    #     event.Hotkey(
    #         "enter",
    #         sync(st.session_state.comand_output),
    #         bindInputs=True,
    #         overrideDefault=True,
    #     )

    #     with w.dashboard(rowHeight=57):
    #         w.editor.update_content("Code", st.session_state.comand_output)

    # with elements("demo"):
    #     event.Hotkey("ctrl+s", sync(), bindInputs=False, overrideDefault=True)

    #     with w.dashboard(rowHeight=57):
    #         w.editor()
# if "comand_output" in st.session_state:
#     st.code(st.session_state.comand_output, language="python")

with col2:
    fig = create_figure()
    if fig is not None:
        st.pyplot(fig=fig)
        fig.savefig(f"image_{st.session_state.number}.jpeg", format="JPEG")
        with open(f"image_{st.session_state.number}.jpeg", "rb") as f:
            st.sidebar.download_button(
                "Download Image", f, file_name=f"image_{st.session_state.number}.jpeg"
            )
        st.sidebar.download_button(
            label="Download Python file",
            data=st.session_state.comand_output,
            file_name=f"python_script_{st.session_state.number}.py",
        )

    # with elements("dashboard"):
    #     # First, build a default layout for every element you want to include in your dashboard
    #     layout = [
    #         # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
    #         dashboard.Item("visualization", 2, 0, 2, 2),
    #     ]

    #     if "comand_output" not in st.session_state:
    #         st.session_state.comand_output = "# The code will be shown in this editor."

    #     def update_content(value):
    #         st.session_state.comand_output = value

    #     # Next, create a dashboard layout using the 'with' syntax. It takes the layout
    #     # as first parameter, plus additional properties you can find in the GitHub links below.
    #     def handle_layout_change(updated_layout):
    #         # You can save the layout in a file, or do anything you want with it.
    #         # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
    #         print(updated_layout)

    #     with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
    #         mui.Paper("first item", key="visualization")

    # If you want to retrieve updated layout values as the user move or resize dashboard items,
    # you can pass a callback to the onLayoutChange event parameter.

    # editor.Monaco(
    #     height=300,
    #     defaultValue=st.session_state.comand_output,
    #     onChange=update_content,
    # )

    # mui.Button("Update content", onClick=sync())

    # editor.MonacoDiff(
    #     original="Happy Streamlit-ing!",
    #     modified="Happy Streamlit-in' with Elements!",
    #     height=300,
    # )


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
