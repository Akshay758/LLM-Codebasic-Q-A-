import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# ==========================================
# 2. Project paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "codebasics_faqs.csv"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_index"
)


# ==========================================
# 3. Gemini LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ==========================================
# 4. Hugging Face Embeddings
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 5. Create Chroma Vector Database
# ==========================================

def create_vector_db():

    print("Loading CSV...")

    loader = CSVLoader(
        file_path=CSV_FILE,
        source_column="prompt"
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    print("Creating Chroma vector database...")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Chroma database created successfully!")
    print(f"Saved at: {CHROMA_PATH}")

    return vectordb


# ==========================================
# 6. Load Chroma Vector Database
# ==========================================

def load_vector_db():

    if not os.path.exists(CHROMA_PATH):

        print("Chroma database not found.")
        print("Creating new database...")

        return create_vector_db()

    print("Loading existing Chroma database...")

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("Chroma database loaded successfully!")

    return vectordb


# ==========================================
# 7. RAG Question Answering
# ==========================================

def ask_question(question):

    # --------------------------------------
    # Load vector database
    # --------------------------------------

    vectordb = load_vector_db()


    # --------------------------------------
    # Create retriever
    # --------------------------------------

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )


    # --------------------------------------
    # Retrieve relevant documents
    # --------------------------------------

    documents = retriever.invoke(question)


    print("\nRetrieved Documents:")
    print("====================")

    for i, doc in enumerate(documents, start=1):

        print(f"\nDocument {i}:")
        print(doc.page_content)


    # --------------------------------------
    # Create context
    # --------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )


    # --------------------------------------
    # Create prompt
    # --------------------------------------

    prompt = ChatPromptTemplate.from_template("""
You are a helpful FAQ assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not available in the context,
say exactly:

"I don't know based on the provided documents."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


    # --------------------------------------
    # Format prompt
    # --------------------------------------

    messages = prompt.format_messages(
        context=context,
        question=question
    )


    # --------------------------------------
    # Ask Gemini
    # --------------------------------------

    response = llm.invoke(messages)


    # --------------------------------------
    # Extract only text from Gemini response
    # --------------------------------------

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    answer += item.get("text", "")

    else:

        answer = response.content


    # --------------------------------------
    # Final answer
    # --------------------------------------

    print("\nFinal Answer:")
    print("=============")
    print(answer)

    return answer


# ==========================================
# 8. Test
# ==========================================

if __name__ == "__main__":

    question = "Do you guys provide internship and also do you offer EMI payments?"

    ask_question(question)