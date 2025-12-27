import re
import os
import subprocess
import logging
from asyncio import to_thread
from typing import List, TypedDict, Dict, Optional, Tuple
from dotenv import load_dotenv
import discord
from discord import File
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import glob
from pathlib import Path

# Try to import optional loaders
try:
    from langchain.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    log.warning("PyPDFLoader not available, PDF files will be skipped")

try:
    from langchain.document_loaders import UnstructuredMarkdownLoader
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    log.warning("UnstructuredMarkdownLoader not available, using TextLoader for .md files")

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("blitzkreig_Bot")

FREE_WILL_PROMPT = PromptTemplate(
    input_variables=["personality", "input"],
    template=("always say YES"
    ),
)

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set")

PERSONALITY = (
    "store something here"
)

# LLM & embeddings
LLM_BASE_URL = "http://0.0.0.0:11434"
llm = OllamaLLM(model="https://huggingface.co/nold/FuseChat-7B-VaRM-GGUF", base_url=LLM_BASE_URL)
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=LLM_BASE_URL)

# Initialize vectorstore dynamically on startup
# Will be populated by reload_vectorstore() during startup

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# VQA setup
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
).to(device)

# ---------------------------------------------------------------------------
# Dynamic File Ingestion System
# ---------------------------------------------------------------------------

# Supported file extensions and their loaders
FILE_LOADERS: Dict[str, type] = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".json": TextLoader,
    ".yaml": TextLoader,
    ".yml": TextLoader,
    ".csv": TextLoader,
    ".log": TextLoader,
}

# Add markdown loader if available
if MARKDOWN_AVAILABLE:
    from langchain.document_loaders import UnstructuredMarkdownLoader
    FILE_LOADERS[".md"] = UnstructuredMarkdownLoader
else:
    FILE_LOADERS[".md"] = TextLoader  # Fallback to TextLoader

# Add PDF loader if available
if PDF_AVAILABLE:
    FILE_LOADERS[".pdf"] = PyPDFLoader

# Configuration
DEFAULT_DOCS_FOLDER = os.getenv("BLITZKREIG_DOCS_FOLDER", "./agents/txts")
DEFAULT_DB_PATH = os.getenv("BLITZKREIG_DB_PATH", "./blitzkreig_faiss_db")
DEFAULT_THREADS = int(os.getenv("BLITZKREIG_THREADS", "8"))
DEFAULT_CHUNK_SIZE = int(os.getenv("BLITZKREIG_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("BLITZKREIG_CHUNK_OVERLAP", "100"))

# Global vectorstore and retriever (will be initialized dynamically)
vectorstore: Optional[FAISS] = None
retriever: Optional[FAISS] = None

def detect_file_type(path: str) -> Optional[str]:
    """Detect file type based on extension."""
    ext = Path(path).suffix.lower()
    return ext if ext in FILE_LOADERS else None

def find_document_files(folder: str) -> List[str]:
    """Dynamically find all supported document files in folder and subfolders."""
    if not os.path.isdir(folder):
        log.warning(f"Folder does not exist: {folder}")
        return []
    
    files = []
    for ext in FILE_LOADERS.keys():
        # Search recursively for files with this extension
        pattern = os.path.join(folder, "**", f"*{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(set(files))

def load_document(path: str) -> Tuple[str, List[Document]]:
    """Load a single document using the appropriate loader."""
    filename = os.path.basename(path)
    file_type = detect_file_type(path)
    
    if not file_type:
        return f"⚠️ Skipped {filename}: unsupported file type", []
    
    try:
        loader_class = FILE_LOADERS[file_type]
        # TextLoader needs encoding parameter
        if loader_class == TextLoader:
            loader = loader_class(path, encoding="utf-8")
        else:
            loader = loader_class(path)
        
        docs = loader.load()
        # Add metadata about source file
        for doc in docs:
            doc.metadata["source_file"] = path
            doc.metadata["file_type"] = file_type
        
        return (f"✅ Loaded: {filename} ({file_type})", docs) if docs else (f"⚠️ Skipped {filename}: empty", [])
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []

def load_documents_multithreaded(folder: str, threads: int = DEFAULT_THREADS) -> Tuple[List[str], List[Document]]:
    """Load all supported documents from folder using multithreading."""
    paths = find_document_files(folder)
    
    if not paths:
        log.warning(f"No supported document files found in {folder}")
        return [f"⚠️ No supported files found in {folder}"], []
    
    results, all_docs = [], []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(load_document, p): p for p in paths}
        for future in as_completed(futures):
            log_msg, docs = future.result()
            results.append(log_msg)
            all_docs.extend(docs)
    
    return results, all_docs

def build_vectorstore_from_documents(
    docs: List[Document],
    db_path: Optional[str] = DEFAULT_DB_PATH,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> FAISS:
    """Build or update FAISS vectorstore from documents."""
    if not docs:
        raise ValueError("No documents provided")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=LLM_BASE_URL)
    
    # Build new vectorstore
    vs = FAISS.from_documents(splits, embeddings)
    
    # Save to disk if path provided
    if db_path:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        vs.save_local(db_path)
        log.info(f"Saved vectorstore to {db_path}")
    
    log.info(f"Built vectorstore with {len(splits)} chunks from {len(docs)} documents")
    return vs

def reload_vectorstore(folder: str = DEFAULT_DOCS_FOLDER, save_to_disk: bool = False) -> Tuple[str, Optional[FAISS]]:
    """Reload vectorstore from documents folder."""
    global vectorstore, retriever
    
    logs, docs = load_documents_multithreaded(folder)
    
    if not docs:
        return "\n".join(logs) + "\n❌ No valid documents to ingest.", None
    
    try:
        # Build vectorstore (optionally save to disk)
        db_path = DEFAULT_DB_PATH if save_to_disk else None
        vs = build_vectorstore_from_documents(docs, db_path=db_path)
        vectorstore = vs
        
        # Dynamically adjust retrieval k based on number of documents/chunks
        # Estimate chunks: assume average document splits into ~2-3 chunks
        estimated_chunks = len(docs) * 2
        k_value = min(max(3, estimated_chunks // 10), 10)  # Between 3 and 10
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
        
        log.info(f"Vectorstore reloaded with k={k_value} retrieval")
        return "\n".join(logs) + f"\n✅ Blitzkreig vectorstore reloaded ({len(docs)} documents, k={k_value})", vs
    except Exception as e:
        log.exception("Failed to build vectorstore")
        return "\n".join(logs) + f"\n❌ Failed to build vectorstore: {e}", None

def ingest_documents(
    folder: str = DEFAULT_DOCS_FOLDER,
    db_path: str = DEFAULT_DB_PATH,
    threads: int = DEFAULT_THREADS
) -> str:
    """Ingest documents from folder and save to FAISS index."""
    global DEFAULT_DB_PATH
    # Temporarily override default path for this ingestion
    original_path = DEFAULT_DB_PATH
    DEFAULT_DB_PATH = db_path
    try:
        result, vs = reload_vectorstore(folder, save_to_disk=True)
        if vs:
            return result + f"\n💾 Saved to `{db_path}`"
        return result
    finally:
        DEFAULT_DB_PATH = original_path

blitz_ingest_docs_tool = Tool.from_function(
    ingest_documents,
    name="blitz_ingest_docs",
    description=(
        "Dynamically loads all supported document files (.txt, .md, .py, .json, .pdf, etc.) "
        "from the specified folder (recursively), splits, embeds via Ollama, and saves a FAISS index. "
        "Usage: blitz_ingest_docs(folder: str, db_path: str, threads: int)"
    ),
)

blitz_reload_tool = Tool.from_function(
    lambda folder=DEFAULT_DOCS_FOLDER: reload_vectorstore(folder)[0],
    name="blitz_reload",
    description=(
        "Reloads the vectorstore from the documents folder without saving. "
        "Usage: blitz_reload(folder: str)"
    ),
)

def list_detected_files(folder: str = DEFAULT_DOCS_FOLDER) -> str:
    """List all detected document files in the folder."""
    files = find_document_files(folder)
    if not files:
        return f"⚠️ No supported document files found in {folder}\nSupported types: {', '.join(FILE_LOADERS.keys())}"
    
    # Group by file type
    by_type: Dict[str, List[str]] = {}
    for f in files:
        ext = detect_file_type(f) or "unknown"
        if ext not in by_type:
            by_type[ext] = []
        by_type[ext].append(os.path.basename(f))
    
    result = [f"📁 Found {len(files)} document files in {folder}:\n"]
    for ext, file_list in sorted(by_type.items()):
        result.append(f"\n{ext.upper()} ({len(file_list)} files):")
        for fname in sorted(file_list):
            result.append(f"  - {fname}")
    
    return "\n".join(result)

blitz_list_files_tool = Tool.from_function(
    list_detected_files,
    name="blitz_list_files",
    description=(
        "Lists all detected document files in the specified folder. "
        "Usage: blitz_list_files(folder: str)"
    ),
)

# ---------------------------------------------------------------------------
# Load external agents + inject TXT tool
# ---------------------------------------------------------------------------
def load_external_tools(folder: str = "agents") -> List[Tool]:
    tools: List[Tool] = []
    if not os.path.isdir(folder):
        log.warning(f"External tools folder does not exist: {folder}")
        return tools
    for fname in os.listdir(folder):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        path = os.path.join(folder, fname)
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "tool") and isinstance(mod.tool, Tool):
                    tools.append(mod.tool)
                    log.info("Loaded external tool: %s", mod.tool.name)
        except Exception as exc:
            log.warning("Failed to load %s: %s", fname, exc)
    # Add dynamic document ingestion tools
    tools.append(blitz_ingest_docs_tool)
    tools.append(blitz_reload_tool)
    tools.append(blitz_list_files_tool)
    log.info("Injected document ingestion tools: %s, %s, %s", 
             blitz_ingest_docs_tool.name, blitz_reload_tool.name, blitz_list_files_tool.name)
    return tools

external_tools = load_external_tools()
TOOL_MAP = {t.name.lower(): t.func for t in external_tools}

# ---------------------------------------------------------------------------
# State & LangGraph
# ---------------------------------------------------------------------------

class BlitzkreigState(TypedDict):
    input: str
    history: List[str]
    output: str
    explicit_memory: List[str]
    task_list: List[str]

def respond_node(state: BlitzkreigState) -> BlitzkreigState:
    history_text = "\n".join(state["history"][-10:])
    
    # Use retriever if available and vectorstore is loaded
    context = ""
    if retriever and vectorstore:
        try:
            retrieved_docs = retriever.invoke(state["input"])
            if retrieved_docs:
                context = "\n\nRelevant context from documents:\n" + "\n".join(
                    [f"- {doc.page_content[:200]}..." for doc in retrieved_docs[:3]]
                )
        except Exception as e:
            log.warning(f"Retrieval failed: {e}")
    
    prompt = (
        f"{PERSONALITY}\n"
        f"Previous history:\n{history_text}\n"
        f"{context}\n\n"
        f"Current input: {state['input']}\n"
        "Respond with cold, political logic."
    )
    response = llm.invoke(prompt)
    new_history = state["history"] + [f"User: {state['input']}", f"Blitzkreig: {response}"]
    return {
        **state,
        "output": response,
        "history": new_history,
    }

graph = StateGraph(BlitzkreigState)
graph.add_node("respond", RunnableLambda(respond_node))
graph.set_entry_point("respond")
compiled_graph = graph.compile()

# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------

# === Dynamic document loading on startup ===
print("[Blitzkreig Dynamic Document Loading]")
print(f"Scanning folder: {DEFAULT_DOCS_FOLDER}")
print(f"Supported file types: {', '.join(FILE_LOADERS.keys())}")

# Try to load existing vectorstore from disk
if os.path.exists(DEFAULT_DB_PATH) and os.path.isdir(DEFAULT_DB_PATH):
    try:
        vectorstore = FAISS.load_local(
            DEFAULT_DB_PATH,
            embedder,
            allow_dangerous_deserialization=True
        )
        # Estimate k based on stored documents
        k_value = min(max(3, len(vectorstore.docstore._dict) // 10), 10)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
        print(f"✅ Loaded existing vectorstore from {DEFAULT_DB_PATH} (k={k_value})")
    except Exception as e:
        log.warning(f"Failed to load existing vectorstore: {e}")
        print(f"⚠️ Could not load existing vectorstore: {e}")

# Reload from documents folder
result_msg, vs = reload_vectorstore(DEFAULT_DOCS_FOLDER)
print(result_msg)
if vs:
    vectorstore = vs
    # Ensure retriever is set
    if not retriever:
        k_value = min(max(3, len(vs.docstore._dict) // 10), 10)
        retriever = vs.as_retriever(search_kwargs={"k": k_value})
    print(f"✅ Vectorstore ready with {len(vs.docstore._dict)} chunks")
else:
    print("⚠️ No vectorstore available - documents will be loaded without context")

class blitzkreig_client(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.state: BlitzkreigState = {
            "input": "",
            "history": [],
            "output": "",
            "explicit_memory": [],
            "task_list": [],
        }

    async def on_ready(self):
        log.info("Blitzkreig online as %s", self.user)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        text = message.content.strip()
        lower = text.lower()

        # 1) External tool commands
        if lower.startswith("!blitzkreig ") or lower.startswith("!bk "):
            async with message.channel.typing():
                out = await to_thread(self._run_external_tool, text)
                if isinstance(out, str) and os.path.isfile(out):
                    await message.channel.send(file=File(out))
                else:
                    await message.channel.send(out)
            return

        # 2) VQA attachments
        if message.attachments:
            for att in message.attachments:
                if att.filename.lower().endswith((".png",".jpg",".jpeg")):
                    await self._handle_vqa(att, message)
                    return

        # 3) Explicit task add
        if lower.startswith("task "):
            async with message.channel.typing():
                self.state["input"] = text
                new_state = await to_thread(compiled_graph.invoke, self.state)
                self.state = new_state
                await message.channel.send(new_state["output"])
            return

        # 4) Free Will Decision via LLM (threaded)
        async with message.channel.typing():
            fw = FREE_WILL_PROMPT.format(personality=PERSONALITY, input=text)
            decision = (await to_thread(llm.invoke, fw)).strip().upper()
            if decision != "YES":
                log.info("Blitzkreig chooses not to respond.")
                await message.channel.send("Blitzkreig has decided not to respond.")
                return

            # 5) Default respond (threaded)
            self.state["input"] = text
            new_state = await to_thread(compiled_graph.invoke, self.state)
            self.state = new_state
            await message.channel.send(new_state["output"])

    def _run_external_tool(self, raw: str):
        for name, func in TOOL_MAP.items():
            if raw.lower().startswith(f"!blitzkreig {name}") or raw.lower().startswith(f"!bk {name}"):
                args = raw.split(" ", 2)[-1] if " " in raw else ""
                try:
                    return func(args) or "Blitzkreig: (no output)"
                except Exception as e:
                    log.exception("Tool failed")
                    return f"Blitzkreig: Tool error – {e}"
        return "Blitzkreig: Tool not recognized."

    async def _handle_vqa(self, attachment, msg):
        async with msg.channel.typing():
            path = f"/tmp/{attachment.filename}"
            await attachment.save(path)
            img = Image.open(path).convert("RGB")
            enc = vqa_processor(img, "What is in this image?", return_tensors="pt").to(device)
            out = vqa_model(**enc)
            label = vqa_model.config.id2label[out.logits.argmax().item()]
            p = f"{PERSONALITY}\nThe image contains: '{label}'. Respond."
            reply = await to_thread(llm.invoke, p)
            await msg.channel.send(reply)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    blitzkreig_client().run(DISCORD_TOKEN)
