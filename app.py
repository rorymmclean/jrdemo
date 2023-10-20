

### Imports
import streamlit as st
import langchain
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import os
import io
import sys
import time
from datetime import datetime
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate


from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain.db")

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain import LLMMathChain
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import json
from contextlib import redirect_stdout
import sqlite3
from sqlite3 import Error
from typing import Optional, Type


### - Layout components --
## I put these at the top because Streamlit runs from the top down and 
## I need a few variables that get defined here. 

## Layout configurations
st.set_page_config(
    page_title='AWS Demo App', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
## CSS is pushed through a markdown configuration.
## As you can probably guess, Streamlit layout is not flexible.
## It's good for internal apps, not so good for customer facing apps.
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

## UI Elements starting with the Top Graphics
col1, col2 = st.columns( [1,5] )
col1.image('AderasBlue2.png', width=50)
col1.image('AderasText.png', width=50)
col2.title('Demo Application for JR')
st.markdown('---')
## Add a sidebar
with st.sidebar: 
    mydemo = st.selectbox('Select Demo', ['Harry Potter', 'SQL Demo'])
    show_detail = st.checkbox('Show Details')
    st.markdown("---")
    tz = st.container()

### --- Housekeeping ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name='gpt-4', openai_api_key=openai_api_key)



### --- Sidebar 1
if mydemo == 'Harry Potter':
    with st.expander("**:blue[Chatting with a document]**"):
        st.markdown("This part of the app lets you make inquiries to \"Harry Potter and the Sorcerer's Stone\". ")
        st.markdown("This might be similar to how you work with product documentation. What I want to show is that it doesn't just dump content from the document it consumed. If you ask a question like \"Who are Harry's best friends?\" it will look at sections of the book dealing with this topic but not just regurgitate them...it will craft a specific answer by interpreting the content of the book in the context of the question you asked.")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask away...?"):
        start = datetime.now()
        tz.write("Start: "+str(start))
        
        pinecone.init(api_key="62fb11b7-e40c-4d06-9ffa-f6db606783b6", environment="gcp-starter")
        embeddings = OpenAIEmbeddings()
        mydocs = Pinecone.from_existing_index(index_name='stone', embedding=embeddings)
        docs = mydocs.similarity_search(
            prompt, 
            k=12)
        
        book_text = ""
        for x in docs:
            book_text = book_text + x.page_content + "\n"

        conversation = ConversationChain(llm=llm, verbose=True)  


        mytemplate=f"""You are a researcher and are helping to answer the users QUESTION below. Use the BOOK_TEXT to answer the question. If the BOOK_TEXT doesn't answer the question respond "I cannot answer your question".

        '''
        BOOK_TEXT:
        {book_text}
        '''

        '''
        QUESTION:
        {prompt}
        '''
        """

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                with st.spinner("Processing..."):
                    response = conversation.run(mytemplate)
        else:
            with st.spinner("Processing..."):
                response = conversation.run(mytemplate)

        st.session_state.messages.append({"role": "assistant", "content": response})    
        st.chat_message('assistant').write(response)

        if show_detail:
            with st.expander('Details', expanded=False):
                s = f.getvalue()
                st.write(s)

        tz.write("End: "+str(datetime.now()))
        tz.write("Duration: "+str(datetime.now() - start))

### --- Sidebar 2
if mydemo == 'SQL Demo':
    with st.expander("**:blue[SQL Querying Demo]**"):
        st.markdown("This demo gives the LLM access to a music database and a calculator. The database has the following tables:")
        st.markdown("""Album (AlbumId, Title, ArtistId)  
Artist (ArtistId, Name)  
Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)  
Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)  
Genre (GenreId, Name)  
Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)  
InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)  
MediaType (MediaTypeId, Name)  
Playlist (PlaylistId, Name)  
PlaylistTrack (PlaylistId, TrackId)  
Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)""") 
        st.markdown("Make up any question like \"what are the top 3 bands in the Rock genre by albums written?\" or \"Show me three sample records from the invoice table\" if you want to learn more about the tables. ChatGPT will do whatever joins are necessary and whatever format you want.")

    ### --- A little housekeeping ---
    ## Creating the SQLite connection information. 
    ## BTW, Streamlit is stateless and all this runs each time the page is drawn.
    conn = None
    try:
        conn = sqlite3.connect('content/chinook.sqlite')
        cur = conn.cursor()
        # st.markdown("connection complete")
    except Error as e:
        print(e)

    ## I need a function that will extract the SQL statement from the response from ChatGPT.
    ## Sometimes it prefices it with strings that deliminates the SQL and sometimes it adds 
    ## an explaination of th code...I just want the code. Prompt engineering works a little but not always.
    def extract_select(text):
        start_index = text.upper().find("SELECT")
        if start_index == -1:
            return None
        end_index = text.find(";", start_index)
        if end_index == -1:
            return None
        sql_statement = text[start_index:end_index + 1]
        return sql_statement.strip()

    ### -- Next up is defining the LangChain chain ---

    ## This is a custom tool. LangChain has predefined tools, but I've had a few problems with it. 
    ## A blogger pointed out that LangChain tools are based upon prompts written for every conceivable
    ## use case. But your own tool will have prompts that are good at your specific use case...so I write my own.
    ## Extended classes isn't really normal Python but that's what you are doing here.
    ## You will see I put the table structure in the description. The description and your question, and 
    ## your prompt all become the prompt submitted to ChatGPT...basically, it's all Prompt Engineering.
    class MySQLTool(BaseTool):
        name = "MySQLTool"
        description = """
    This tool queries a SQLite database. It is useful for when you need to answer questions 
    by running SQLite queries. Always indicate if your response is a "thought" or a "final answer". 
    The following TABLES information is provided to help you write your sql statement.
    Be sure to end all SQL statements with a semicolon. 
            
    TABLES:
    Album (AlbumId, Title, ArtistId)
    Artist (ArtistId, Name)
    Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
    Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
    Genre (GenreId, Name)
    Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
    InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
    MediaType (MediaTypeId, Name)
    Playlist (PlaylistId, Name)
    PlaylistTrack (PlaylistId, TrackId)
    Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
    """

        def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool."""

            # strip it down to the SELECT statement
            newquery = extract_select(query)

            ### Uncomment if you want the query printed on the page
            # st.markdown("#### Query Being Executed:")
            # st.markdown(newquery)

            # I previously defined the cursor
            try:
                cur.execute(newquery)
                results = cur.fetchall()
                # st.markdown(results)
            except Error as e:
                results = "Error running query"
            
            return results  
        
        # I'm not using async but you could if you want the prompts to update 
        # the user screen as it progresses.
        async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("custom_search does not support async")

    ## Besides SQL, I defined an agent that can perform mathematics.
    ## I'm just tossing this in to demonstrate chains are more than one tool.
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=False)
    # palchain = PALChain.from_math_prompt(llm=llm, verbose=True)

    ## Now we define the tools that will be used in the chain. They look different
    ## because our custom tool already has thse properties defined in the package.
    tools = [
        Tool(
            name="Calculator",
            func=llm_math.run,
            description="This tool is good at solving complex word math problems. Input should be a fully worded hard word math problem."

    # Use the following format:"
        ),
        MySQLTool()]

    ## Now we define the agent
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=False,
    )

    ## One more package...this runs when you enter a question and hit [ENTER]
    ## It runs everything and then finishes paining the rest of the page
    def run_prompt(myquestion):
        # I'm using a chat approach so the questions and answers scroll down the page as you use the app.
        # It handles all the writing that occurs after you hit enter.
        st.session_state.messages.append({"role": "user", "content": myquestion})
        st.chat_message("user").write(myquestion)

        # I probably didn't need to put the database info in another prompt.
        template=f"""You are a DBA helping the user write and run ANSI SQL queries.
        Using the DATABASE INFORMATION below, provide a SQL query to the "QUESTION" below. 
        Be sure to end all SQL statements with a semicolon.

        DATABASE INFORMATION:
        Album (AlbumId, Title, ArtistId)
        Artist (ArtistId, Name)
        Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
        Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
        Genre (GenreId, Name)
        Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
        InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
        MediaType (MediaTypeId, Name)
        Playlist (PlaylistId, Name)
        PlaylistTrack (PlaylistId, TrackId)
        Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)

        QUESTION:
        {myquestion}
        """

        # If detail is requested it captures the stdout.
        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                with st.spinner("Processing..."):
                    response = search_agent.run(template)
        else:
            with st.spinner("Processing..."):
                response = search_agent.run(template)
        
        st.session_state.messages.append({"role": "assistant", "content": response})    
        st.chat_message('assistant').write(response)

        # If detail is requested the stdout is printed it in a collapsable region.
        if show_detail:
            with st.expander('Details', expanded=False):
                s = f.getvalue()
                st.write(s)

    ### -- Let's get back to building the web page --
    ## First run populates the session state with a benign message
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    ## History Q&As are printed
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    ## Now we ask for a question
    if prompt := st.chat_input(placeholder="Ask a query-like question?"):
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_prompt(prompt)
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))


