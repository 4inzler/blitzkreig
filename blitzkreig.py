import re
import os
import subprocess
import logging
from datetime import datetime
from asyncio import to_thread
from typing import List, TypedDict, Dict, Any
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
# TXT Ingestion Tool - Dynamic File Discovery & Loading
# ---------------------------------------------------------------------------
TXT_FOLDER      = os.getenv("BLITZKRIEG_TXT_FOLDER", "./agents/txts")
DEFAULT_DB_PATH = os.getenv("BLITZKRIEG_DB_PATH", "./blitzkreig_faiss_db")
DEFAULT_THREADS = int(os.getenv("BLITZKRIEG_THREADS", "8"))

# Track loaded files metadata
loaded_files_metadata = []

def load_txt(path: str):
    filename = os.path.basename(path)
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        if docs:
            file_size = os.path.getsize(path)
            mod_time = os.path.getmtime(path)
            char_count = sum(len(doc.page_content) for doc in docs)
            metadata = {
                "filename": filename,
                "path": path,
                "size": file_size,
                "modified": mod_time,
                "char_count": char_count,
                "doc_count": len(docs)
            }
            loaded_files_metadata.append(metadata)
            return (f"✅ Loaded: {filename} ({file_size} bytes, {char_count} chars, {len(docs)} docs)", docs)
        else:
            return (f"⚠️ Skipped {filename}: empty", [])
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []

def load_txts_multithreaded(folder: str, threads: int):
    results, all_docs = [], []
    loaded_files_metadata.clear()  # Reset metadata on each load
    
    if not os.path.exists(folder):
        log.warning(f"TXT folder does not exist: {folder}")
        return [f"❌ Folder not found: {folder}"], []
    
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
    
    if not paths:
        return [f"⚠️ No .txt files found in {folder}"], []
    
    log.info(f"Discovering {len(paths)} .txt files in {folder}...")
    
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(load_txt, p): p for p in paths}
        for future in as_completed(futures):
            log_msg, docs = future.result()
            results.append(log_msg)
            all_docs.extend(docs)
    
    return results, all_docs

def get_loaded_files_summary() -> str:
    if not loaded_files_metadata:
        return "No files currently loaded."
    
    summary = [
        f"📚 **Blitzkrieg Knowledge Base Summary**",
        f"Total files: {len(loaded_files_metadata)}",
        f"Total documents: {sum(m['doc_count'] for m in loaded_files_metadata)}",
        f"Total characters: {sum(m['char_count'] for m in loaded_files_metadata):,}",
        f"Total size: {sum(m['size'] for m in loaded_files_metadata):,} bytes",
        "",
        "**Loaded Files:**"
    ]
    
    for meta in sorted(loaded_files_metadata, key=lambda x: x['filename']):
        mod_date = datetime.fromtimestamp(meta['modified']).strftime('%Y-%m-%d %H:%M:%S')
        summary.append(
            f"  • {meta['filename']}: {meta['char_count']:,} chars, {meta['doc_count']} docs (modified: {mod_date})"
        )
    
    return "\n".join(summary)

def ingest_txts(
    folder: str = TXT_FOLDER,
    db_path: str = DEFAULT_DB_PATH,
    threads: int = DEFAULT_THREADS
) -> str:
    log.info(f"🔄 Starting TXT ingestion from: {folder}")
    logs, docs = load_txts_multithreaded(folder, threads)
    if not docs:
        return "\n".join(logs) + "\n❌ No valid text docs to ingest."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    log.info(f"✂️ Split {len(docs)} documents into {len(splits)} chunks")

    embeddings = OllamaEmbeddings(model="mistral", base_url=LLM_BASE_URL)
    log.info(f"🧠 Creating embeddings and building FAISS index...")
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(db_path)
    log.info(f"💾 Saved FAISS index to: {db_path}")

    summary = get_loaded_files_summary()
    return "\n".join(logs) + f"\n\n✅ Blitzkreig FAISS store saved to `{db_path}`\n\n{summary}"

blitz_ingest_txts_tool = Tool.from_function(
    ingest_txts,
    name="blitz_ingest_txts",
    description=(
        "Loads all .txt files from Blitzkreig's txts folder, splits, embeds via Ollama, "
        "and saves a FAISS index. Usage: blitz_ingest_txts(db_path: str, threads: int)"
    ),
)

blitz_show_files_tool = Tool.from_function(
    get_loaded_files_summary,
    name="blitz_show_files",
    description=(
        "Shows a summary of all currently loaded .txt files including file names, "
        "character counts, and document counts. Usage: blitz_show_files()"
    ),
)

blitz_reload_tool = Tool.from_function(
    lambda: ingest_txts(TXT_FOLDER, DEFAULT_DB_PATH, DEFAULT_THREADS),
    name="blitz_reload",
    description=(
        "Reloads all .txt files from the configured folder and rebuilds the FAISS index. "
        "Usage: blitz_reload()"
    ),
)

# ---------------------------------------------------------------------------
# Load external agents + inject TXT tool
# ---------------------------------------------------------------------------
def load_external_tools(folder: str = "agents") -> List[Tool]:
    tools: List[Tool] = []
    if not os.path.isdir(folder):
        log.warning(f"Agent folder not found: {folder}")
        return tools
    
    log.info(f"🔍 Scanning for external tools in: {folder}")
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
                    log.info(f"✅ Loaded external tool: {mod.tool.name}")
        except Exception as exc:
            log.warning(f"⚠️ Failed to load {fname}: {exc}")
    
    # Inject built-in Blitzkrieg tools
    tools.extend([blitz_ingest_txts_tool, blitz_show_files_tool, blitz_reload_tool])
    log.info(f"✅ Injected {len([blitz_ingest_txts_tool, blitz_show_files_tool, blitz_reload_tool])} Blitzkrieg built-in tools")
    log.info(f"📦 Total tools available: {len(tools)}")
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

# === TXT folder pre-load on startup ===
log.info(f"🚀 Blitzkrieg Bot Starting Up...")
log.info(f"📂 TXT Folder: {TXT_FOLDER}")
log.info(f"💾 Database Path: {DEFAULT_DB_PATH}")
log.info(f"🧵 Thread Count: {DEFAULT_THREADS}")

logs, docs = load_txts_multithreaded(TXT_FOLDER, DEFAULT_THREADS)
print("\n" + "="*70)
print("[📚 Blitzkreig Knowledge Base Loading Results]")
print("="*70)
for logline in logs:
    print(logline)
print(f"\n📊 Total loaded documents: {len(docs)}")
if loaded_files_metadata:
    print(f"📁 Files loaded: {len(loaded_files_metadata)}")
    print(f"📝 Total characters: {sum(m['char_count'] for m in loaded_files_metadata):,}")
print("="*70 + "\n")

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

        # 1) Built-in info commands
        if lower in ["!blitzkreig files", "!bk files"]:
            async with message.channel.typing():
                summary = get_loaded_files_summary()
                await message.channel.send(summary)
            return
        
        if lower in ["!blitzkreig reload", "!bk reload"]:
            async with message.channel.typing():
                result = await to_thread(ingest_txts, TXT_FOLDER, DEFAULT_DB_PATH, DEFAULT_THREADS)
                # Split long messages for Discord
                for i in range(0, len(result), 2000):
                    await message.channel.send(result[i:i+2000])
            return
        
        if lower in ["!blitzkreig help", "!bk help"]:
            help_text = (
                "**🔧 Blitzkrieg Bot Commands**\n"
                "• `!blitzkreig files` or `!bk files` - Show loaded knowledge base files\n"
                "• `!blitzkreig reload` or `!bk reload` - Reload all .txt files and rebuild index\n"
                "• `!blitzkreig help` or `!bk help` - Show this help message\n"
                f"• `!blitzkreig <tool_name>` - Run external tool (Available: {', '.join(TOOL_MAP.keys())})\n"
                "• Send an image - Visual Question Answering\n"
                "• Normal message - Chat with Blitzkrieg\n"
            )
            await message.channel.send(help_text)
            return

        # 2) External tool commands
        if lower.startswith("!blitzkreig ") or lower.startswith("!bk "):
            async with message.channel.typing():
                out = await to_thread(self._run_external_tool, text)
                if isinstance(out, str) and os.path.isfile(out):
                    await message.channel.send(file=File(out))
                else:
                    # Split long messages for Discord
                    for i in range(0, len(out), 2000):
                        await message.channel.send(out[i:i+2000])
            return

        # 3) VQA attachments
        if message.attachments:
            for att in message.attachments:
                if att.filename.lower().endswith((".png",".jpg",".jpeg")):
                    await self._handle_vqa(att, message)
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
            # Split long messages for Discord
            response = new_state["output"]
            for i in range(0, len(response), 2000):
                await message.channel.send(response[i:i+2000])

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
