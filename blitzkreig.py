import re
import os
import subprocess
import logging
import mimetypes
from asyncio import to_thread
from typing import List, TypedDict, Dict, Callable, Optional, Any
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

vectorstore = FAISS.from_texts(
    [
        "store something here"
    ],
    embedder,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# VQA setup
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
).to(device)

# ---------------------------------------------------------------------------
# Dynamic File Handler
# ---------------------------------------------------------------------------
DOCS_FOLDER     = "./agents/docs"
DEFAULT_DB_PATH = "./blitzkreig_faiss_db"
DEFAULT_THREADS = 8

# Supported file extensions and their categories
FILE_CATEGORIES: Dict[str, List[str]] = {
    "text": [".txt", ".md", ".rst", ".log"],
    "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb", ".php", ".sh", ".bash"],
    "data": [".json", ".csv", ".xml", ".yaml", ".yml", ".toml"],
    "document": [".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
}

# Flatten for quick lookup
SUPPORTED_EXTENSIONS: Dict[str, str] = {}
for category, extensions in FILE_CATEGORIES.items():
    for ext in extensions:
        SUPPORTED_EXTENSIONS[ext] = category


def get_file_category(filename: str) -> Optional[str]:
    """Determine the category of a file based on its extension."""
    ext = os.path.splitext(filename.lower())[1]
    return SUPPORTED_EXTENSIONS.get(ext)


def is_supported_file(filename: str) -> bool:
    """Check if a file is supported for processing."""
    return get_file_category(filename) is not None


def load_text_file(path: str) -> tuple:
    """Load a text-based file (txt, md, code, etc.)."""
    filename = os.path.basename(path)
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        return (f"✅ Loaded: {filename}", docs) if docs else (f"⚠️ Skipped {filename}: empty", [])
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []


def load_json_file(path: str) -> tuple:
    """Load and parse a JSON file."""
    import json
    filename = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert JSON to readable text for embedding
        text_content = json.dumps(data, indent=2)
        from langchain.schema import Document
        doc = Document(page_content=text_content, metadata={"source": path, "type": "json"})
        return f"✅ Loaded JSON: {filename}", [doc]
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []


def load_csv_file(path: str) -> tuple:
    """Load and parse a CSV file."""
    import csv
    filename = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Convert CSV to readable text
        text_content = "\n".join([", ".join(row) for row in rows])
        from langchain.schema import Document
        doc = Document(page_content=text_content, metadata={"source": path, "type": "csv"})
        return f"✅ Loaded CSV: {filename}", [doc]
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []


def load_pdf_file(path: str) -> tuple:
    """Load a PDF file if PyPDF is available."""
    filename = os.path.basename(path)
    try:
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        docs = loader.load()
        return (f"✅ Loaded PDF: {filename}", docs) if docs else (f"⚠️ Skipped {filename}: empty", [])
    except ImportError:
        return f"⚠️ Skipped {filename}: PyPDF not installed", []
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []


# Dynamic loader mapping based on file extension
FILE_LOADERS: Dict[str, Callable] = {
    # Text files
    ".txt": load_text_file,
    ".md": load_text_file,
    ".rst": load_text_file,
    ".log": load_text_file,
    # Code files (treat as text)
    ".py": load_text_file,
    ".js": load_text_file,
    ".ts": load_text_file,
    ".java": load_text_file,
    ".cpp": load_text_file,
    ".c": load_text_file,
    ".h": load_text_file,
    ".go": load_text_file,
    ".rs": load_text_file,
    ".rb": load_text_file,
    ".php": load_text_file,
    ".sh": load_text_file,
    ".bash": load_text_file,
    # Data files
    ".json": load_json_file,
    ".csv": load_csv_file,
    ".xml": load_text_file,
    ".yaml": load_text_file,
    ".yml": load_text_file,
    ".toml": load_text_file,
    # Documents
    ".pdf": load_pdf_file,
}


def load_file_dynamic(path: str) -> tuple:
    """Dynamically load a file based on its extension."""
    ext = os.path.splitext(path.lower())[1]
    loader_func = FILE_LOADERS.get(ext, load_text_file)
    return loader_func(path)


def register_file_type(extension: str, category: str, loader: Optional[Callable] = None) -> str:
    """
    Register a new file type for dynamic handling.
    
    Args:
        extension: File extension (e.g., ".html")
        category: Category ("text", "code", "data", "document", "image")
        loader: Optional custom loader function. Defaults to text loader.
    
    Returns:
        Status message
    """
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    
    if ext in SUPPORTED_EXTENSIONS:
        return f"⚠️ Extension {ext} already registered as {SUPPORTED_EXTENSIONS[ext]}"
    
    SUPPORTED_EXTENSIONS[ext] = category
    if category in FILE_CATEGORIES:
        FILE_CATEGORIES[category].append(ext)
    else:
        FILE_CATEGORIES[category] = [ext]
    
    FILE_LOADERS[ext] = loader or load_text_file
    
    return f"✅ Registered {ext} as {category}"


def get_supported_formats() -> str:
    """Get a formatted string of all supported file formats by category."""
    lines = ["📁 **Supported File Formats:**"]
    for category, extensions in FILE_CATEGORIES.items():
        exts = ", ".join(sorted(extensions))
        lines.append(f"  • **{category.capitalize()}**: {exts}")
    return "\n".join(lines)


def get_supported_files(folder: str) -> List[str]:
    """Get all supported files from a folder."""
    if not os.path.isdir(folder):
        return []
    files = []
    for f in os.listdir(folder):
        if is_supported_file(f):
            files.append(os.path.join(folder, f))
    return files


def load_files_multithreaded(folder: str, threads: int, extensions: Optional[List[str]] = None):
    """Load multiple files in parallel, optionally filtering by extensions."""
    results, all_docs = [], []
    
    if extensions:
        # Filter by specific extensions
        paths = [
            os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if os.path.splitext(f.lower())[1] in extensions
        ]
    else:
        # Load all supported files
        paths = get_supported_files(folder)
    
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(load_file_dynamic, p): p for p in paths}
        for future in as_completed(futures):
            log_msg, docs = future.result()
            results.append(log_msg)
            all_docs.extend(docs)
    return results, all_docs


def ingest_files(
    folder: str = DOCS_FOLDER,
    db_path: str = DEFAULT_DB_PATH,
    threads: int = DEFAULT_THREADS,
    extensions: Optional[List[str]] = None
) -> str:
    """Ingest files from a folder into a FAISS vector store."""
    logs, docs = load_files_multithreaded(folder, threads, extensions)
    if not docs:
        return "\n".join(logs) + "\n❌ No valid documents to ingest."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=LLM_BASE_URL)
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(db_path)

    supported_exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
    return "\n".join(logs) + f"\n✅ Blitzkreig FAISS store saved to `{db_path}`\n📁 Supported formats: {supported_exts}"


blitz_ingest_files_tool = Tool.from_function(
    ingest_files,
    name="blitz_ingest_files",
    description=(
        "Dynamically loads files from Blitzkreig's docs folder based on file type. "
        "Supports: .txt, .md, .py, .js, .json, .csv, .pdf, and more. "
        "Usage: blitz_ingest_files(folder: str, db_path: str, threads: int)"
    ),
)

# ---------------------------------------------------------------------------
# Load external agents + inject dynamic file tool
# ---------------------------------------------------------------------------
def load_external_tools(folder: str = "agents") -> List[Tool]:
    tools: List[Tool] = []
    if not os.path.isdir(folder):
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
    tools.append(blitz_ingest_files_tool)
    log.info("Injected dynamic file ingestion tool: %s", blitz_ingest_files_tool.name)
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
    prompt = (
        f"{PERSONALITY}\n"
        f"Previous history:\n{history_text}\n\n"
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

# === Dynamic file pre-load on startup ===
if os.path.isdir(DOCS_FOLDER):
    logs, docs = load_files_multithreaded(DOCS_FOLDER, DEFAULT_THREADS)
    print("[Blitzkreig Dynamic File Load Results]")
    for logline in logs:
        print(logline)
    print(f"Total loaded documents: {len(docs)}")
    print(f"Supported file types: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}\n")
else:
    docs = []
    print(f"[Blitzkreig] Docs folder '{DOCS_FOLDER}' not found. Skipping pre-load.\n")

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

        # 1) Show supported formats
        if lower in ("!blitzkreig formats", "!bk formats", "!blitzkreig filetypes", "!bk filetypes"):
            await message.channel.send(get_supported_formats())
            return

        # 2) External tool commands
        if lower.startswith("!blitzkreig ") or lower.startswith("!bk "):
            async with message.channel.typing():
                out = await to_thread(self._run_external_tool, text)
                if isinstance(out, str) and os.path.isfile(out):
                    await message.channel.send(file=File(out))
                else:
                    await message.channel.send(out)
            return

        # 3) Dynamic file attachment handling
        if message.attachments:
            for att in message.attachments:
                file_category = get_file_category(att.filename)
                if file_category == "image":
                    await self._handle_vqa(att, message)
                    return
                elif file_category in ("text", "code", "data", "document"):
                    await self._handle_document(att, message, file_category)
                    return
                else:
                    # Unknown file type - inform user of supported types
                    await message.channel.send(
                        f"Blitzkreig: Unsupported file type.\n{get_supported_formats()}"
                    )
                    return

        # 4) Explicit task add
        if lower.startswith("task "):
            async with message.channel.typing():
                self.state["input"] = text
                new_state = await to_thread(compiled_graph.invoke, self.state)
                self.state = new_state
                await message.channel.send(new_state["output"])
            return

        # 5) Free Will Decision via LLM (threaded)
        async with message.channel.typing():
            fw = FREE_WILL_PROMPT.format(personality=PERSONALITY, input=text)
            decision = (await to_thread(llm.invoke, fw)).strip().upper()
            if decision != "YES":
                log.info("Blitzkreig chooses not to respond.")
                await message.channel.send("Blitzkreig has decided not to respond.")
                return

            # 6) Default respond (threaded)
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
        """Handle image attachments with Visual Question Answering."""
        async with msg.channel.typing():
            path = f"/tmp/{attachment.filename}"
            await attachment.save(path)
            try:
                img = Image.open(path).convert("RGB")
                enc = vqa_processor(img, "What is in this image?", return_tensors="pt").to(device)
                out = vqa_model(**enc)
                label = vqa_model.config.id2label[out.logits.argmax().item()]
                p = f"{PERSONALITY}\nThe image contains: '{label}'. Respond."
                reply = await to_thread(llm.invoke, p)
                await msg.channel.send(reply)
            except Exception as e:
                log.exception("VQA failed for %s", attachment.filename)
                await msg.channel.send(f"Blitzkreig: Failed to analyze image – {e}")
            finally:
                if os.path.exists(path):
                    os.remove(path)

    async def _handle_document(self, attachment, msg, file_category: str):
        """Handle text, code, data, and document attachments dynamically."""
        async with msg.channel.typing():
            path = f"/tmp/{attachment.filename}"
            await attachment.save(path)
            try:
                # Use the dynamic file loader
                log_msg, docs = await to_thread(load_file_dynamic, path)
                
                if not docs:
                    await msg.channel.send(f"Blitzkreig: Could not process file. {log_msg}")
                    return
                
                # Extract content from documents
                content = "\n\n".join([doc.page_content[:2000] for doc in docs[:3]])  # Limit content
                
                # Create a contextual prompt based on file type
                ext = os.path.splitext(attachment.filename)[1].lower()
                file_type_desc = {
                    "text": "text document",
                    "code": f"source code ({ext})",
                    "data": f"data file ({ext})",
                    "document": "document",
                }.get(file_category, "file")
                
                prompt = (
                    f"{PERSONALITY}\n\n"
                    f"The user has shared a {file_type_desc} named '{attachment.filename}'.\n"
                    f"Content preview:\n```\n{content[:1500]}\n```\n\n"
                    f"Analyze and respond to this {file_type_desc}."
                )
                
                reply = await to_thread(llm.invoke, prompt)
                
                # Chunk response if too long
                for i in range(0, len(reply), 2000):
                    await msg.channel.send(reply[i:i+2000])
                    
            except Exception as e:
                log.exception("Document processing failed for %s", attachment.filename)
                await msg.channel.send(f"Blitzkreig: Failed to process file – {e}")
            finally:
                if os.path.exists(path):
                    os.remove(path)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    blitzkreig_client().run(DISCORD_TOKEN)
