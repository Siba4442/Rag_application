import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rag_class_loader import DocLoader
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import time  # Add time for unique key generation

# Load environment variables
load_dotenv()

# Textsplitter configuration
sep = "\n"
chunk_size = 1000
chunk_overlap = 100

# Huggingface text embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}

embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

collection_name = os.getenv('CHROMA_COLLECTION_NAME')

# Streamlit Application
st.title("AI-Powered Chatbot with Document Upload")


# Sidebar: Ask for GROQ API Key
if "GROQ_API_KEY" not in st.session_state:
    st.sidebar.title("Boot Section")
    groq_api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")
    if groq_api_key:
        st.session_state["GROQ_API_KEY"] = groq_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.sidebar.success("GROQ API Key set successfully!")


# Step 1: Ask for PDF(s) upload
if "vector_store" not in st.session_state:
    uploaded_files = st.file_uploader("Upload PDF(s) for analysis:", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        pdf_directory = os.getenv('PDF_DIRECTORY')
        os.makedirs(pdf_directory, exist_ok=True)

        for uploaded_file in uploaded_files:
            with open(os.path.join(pdf_directory, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

        st.success("PDF(s) uploaded successfully!")

        # Initialize DocLoader
        docloader = DocLoader(
            pdf_directory,
            os.getenv('CLIENT_TYPE'),
            os.getenv('VECTORDB_DIRECTORY'),
            collection_name,
            sep,
            chunk_size,
            chunk_overlap,
            embedding_function
        )

        llm = ChatGroq(model="llama3-8b-8192")
        docloader.create_update_vectorstore()
        st.session_state["vector_store"] = docloader.get_vector_store()
        st.success("Vector store created and loaded successfully! You can now ask questions.")

if "vector_store" in st.session_state:
    vector_store = st.session_state["vector_store"]

    # Initialize graph only once and keep it available
    if "graph" not in st.session_state:
        # Define tools and build the graph
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = llm.bind_tools([retrieve])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        tools = ToolNode([retrieve])

        def generate(state: MessagesState):
            """Generate answer."""
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages

            response = llm.invoke(prompt)
            return {"messages": [response]}

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        memory = MemorySaver()
        st.session_state["graph"] = graph_builder.compile(checkpointer=memory)

        # Ensure the config is retained
        if "config" not in st.session_state:
            st.session_state["config"] = {"configurable": {"thread_id": "abc123"}}

    # Retrieve the graph from session state
    graph = st.session_state["graph"]
    config = st.session_state["config"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message_pair in st.session_state.chat_history:
        with st.chat_message(message_pair["role"]):
            st.markdown(message_pair["content"])

    # Handle user input and generate response
    user_input = st.chat_input("Ask AI...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Generate response using the graph.stream()
            response_message = None
            for step in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
                config=config,
            ):
                response_message = step

            assistant_response = response_message["messages"][-1].content
            st.markdown(assistant_response)

            # Add assistant's response to history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
