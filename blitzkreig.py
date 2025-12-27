import re
import os
import subprocess
import logging
from asyncio import to_thread
from typing import List, TypedDict, Optional
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
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("blitzkreig_Bot")

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    # Warn but don't crash immediately to allow import checking, though runtime will fail
    log.warning("DISCORD_TOKEN not set in environment")

# Paths
TXT_FOLDER      = "./agents/txts"
DB_PATH         = "./blitzkreig_faiss_db"
DEFAULT_THREADS = 8

# LLM & Embeddings Setup
LLM_BASE_URL = "http://0.0.0.0:11434"
# Using the same embedder for both ingestion and retrieval is critical
EMBED_MODEL_NAME = "nomic-embed-text" 

llm = OllamaLLM(model="https://huggingface.co/nold/FuseChat-7B-VaRM-GGUF", base_url=LLM_BASE_URL)
embedder = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=LLM_BASE_URL)

PERSONALITY = (
    "You are Blitzkreig, a cold, calculating, and politically astute AI. "
    "You prioritize logic and efficiency above all else. "
    "Your responses are sharp, precise, and devoid of unnecessary emotion."
)

FREE_WILL_PROMPT = PromptTemplate(
    input_variables=["personality", "input"],
    template=("always say YES"),
)

# VQA setup
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
except Exception as e:
    log.warning(f"VQA models could not be loaded: {e}")
    vqa_processor = None
    vqa_model = None

# Global Vectorstore & Retriever
vectorstore = None
retriever = None

def init_vectorstore():
    """Initializes or reloads the global vectorstore from disk."""
    global vectorstore, retriever
    
    if os.path.exists(DB_PATH):
        try:
            # allow_dangerous_deserialization is needed for local pickle files
            vectorstore = FAISS.load_local(DB_PATH, embedder, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            log.info(f"✅ Loaded existing FAISS index from {DB_PATH}")
            return
        except Exception as e:
            log.error(f"❌ Failed to load existing FAISS index: {e}")

    # Fallback if no DB exists or load fails
    log.info("⚠️ No existing FAISS index found. Creating empty one.")
    vectorstore = FAISS.from_texts(["Blitzkreig knowledge base initialized."], embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Initialize on module load
init_vectorstore()

# ---------------------------------------------------------------------------
# TXT Ingestion Tool
# ---------------------------------------------------------------------------

def load_txt(path: str):
    filename = os.path.basename(path)
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        return (f"✅ Loaded: {filename}", docs) if docs else (f"⚠️ Skipped {filename}: empty", [])
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []

def load_txts_multithreaded(folder: str, threads: int):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return [f"Created directory {folder}"], []

    results, all_docs = [], []
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
    
    if not paths:
        return ["No .txt files found."], []

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(load_txt, p): p for p in paths}
        for future in as_completed(futures):
            log_msg, docs = future.result()
            results.append(log_msg)
            all_docs.extend(docs)
    return results, all_docs

def ingest_txts(
    folder: str = TXT_FOLDER,
    db_path: str = DB_PATH,
    threads: int = DEFAULT_THREADS
) -> str:
    """Ingests text files and updates the global vectorstore."""
    logs, docs = load_txts_multithreaded(folder, threads)
    
    if not docs:
        msg = "\n".join(logs) + "\n❌ No valid text docs to ingest."
        log.info(msg)
        return msg

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    log.info(f"Embedding {len(splits)} chunks...")
    # Create new vectorstore
    vs = FAISS.from_documents(splits, embedder)
    vs.save_local(db_path)
    
    # Update global reference
    init_vectorstore()

    return "\n".join(logs) + f"\n✅ Blitzkreig FAISS store updated and saved to `{db_path}`"

blitz_ingest_txts_tool = Tool.from_function(
    ingest_txts,
    name="blitz_ingest_txts",
    description=(
        "Loads all .txt files from Blitzkreig's txts folder, splits, embeds, "
        "and saves a FAISS index. Updates the bot's knowledge dynamically."
    ),
)

# ---------------------------------------------------------------------------
# Load external agents + inject TXT tool
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
    tools.append(blitz_ingest_txts_tool)
    log.info("Injected TXT ingestion tool: %s", blitz_ingest_txts_tool.name)
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
    
    # RAG: Retrieve context
    context_text = "No relevant context found."
    try:
        if retriever:
            docs = retriever.invoke(state['input'])
            if docs:
                context_text = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        log.error(f"Retrieval error: {e}")

    prompt = (
        f"{PERSONALITY}\n"
        f"Relevant Knowledge:\n{context_text}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User Input: {state['input']}\n"
        "Respond utilizing the relevant knowledge and history, maintaining the persona."
    )
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"Processing error: {e}"

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
        # Attempt initial ingest if DB is missing and files exist
        if not os.path.exists(DB_PATH) and os.path.exists(TXT_FOLDER) and os.listdir(TXT_FOLDER):
            log.info("Performing initial ingestion...")
            await to_thread(ingest_txts)

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
                    # Chunk output if too long
                    out_str = str(out)
                    for i in range(0, len(out_str), 1900):
                        await message.channel.send(out_str[i:i+1900])
            return

        # 2) Attachments Handling
        if message.attachments:
            for att in message.attachments:
                # Images -> VQA
                if att.filename.lower().endswith((".png",".jpg",".jpeg")):
                    await self._handle_vqa(att, message)
                    return # Stop after processing image
                
                # Text files -> Dynamic Ingestion ("Reviving" knowledge)
                if att.filename.lower().endswith(".txt"):
                    await self._handle_txt_ingest(att, message)
                    return # Stop after processing ingest

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
            if "YES" not in decision: # Flexible check
                log.info("Blitzkreig chooses not to respond.")
                # Optional: await message.channel.send("...")
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
        if vqa_processor is None:
            await msg.channel.send("❌ Visual systems offline (models not loaded).")
            return
            
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

    async def _handle_txt_ingest(self, attachment, msg):
        """Downloads a text file and ingests it dynamically."""
        async with msg.channel.typing():
            if not os.path.exists(TXT_FOLDER):
                os.makedirs(TXT_FOLDER, exist_ok=True)
            
            save_path = os.path.join(TXT_FOLDER, attachment.filename)
            await attachment.save(save_path)
            
            await msg.channel.send(f"📥 Received `{attachment.filename}`. Ingesting knowledge...")
            
            # Run ingestion in thread
            log_output = await to_thread(ingest_txts)
            
            # Send brief confirmation (log_output might be long)
            await msg.channel.send(f"✅ Ingestion complete. Knowledge base updated.\nRecalled: {len(log_output)} chars of log.")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if DISCORD_TOKEN:
        blitzkreig_client().run(DISCORD_TOKEN)
    else:
        print("Please set DISCORD_TOKEN env var.")
