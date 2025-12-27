import re
import os
import json
import subprocess
import logging
from asyncio import to_thread
from pathlib import Path
from typing import Any, List, TypedDict
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
# TXT Ingestion Tool
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
TXT_FOLDER = os.getenv("BLITZ_TXT_FOLDER", str(PROJECT_ROOT / "agents" / "txts"))
DEFAULT_DB_PATH = "./blitzkreig_faiss_db"
DEFAULT_THREADS = 8

def load_txt(path: str):
    filename = os.path.basename(path)
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        return (f"✅ Loaded: {filename}", docs) if docs else (f"⚠️ Skipped {filename}: empty", [])
    except Exception as e:
        return f"❌ Error in {filename}: {e}", []

def load_txts_multithreaded(folder: str, threads: int):
    results, all_docs = [], []
    if not os.path.isdir(folder):
        return [f"❌ TXT folder not found: {folder}"], []
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(load_txt, p): p for p in paths}
        for future in as_completed(futures):
            log_msg, docs = future.result()
            results.append(log_msg)
            all_docs.extend(docs)
    return results, all_docs

def ingest_txts(
    folder: str = TXT_FOLDER,
    db_path: str = DEFAULT_DB_PATH,
    threads: int = DEFAULT_THREADS
) -> str:
    logs, docs = load_txts_multithreaded(folder, threads)
    if not docs:
        return "\n".join(logs) + "\n❌ No valid text docs to ingest."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mistral")
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(db_path)

    return "\n".join(logs) + f"\n✅ Blitzkreig FAISS store saved to `{db_path}`"

blitz_ingest_txts_tool = Tool.from_function(
    ingest_txts,
    name="blitz_ingest_txts",
    description=(
        "Loads all .txt files from Blitzkreig's txts folder, splits, embeds via Ollama, "
        "and saves a FAISS index. Usage: blitz_ingest_txts(db_path: str, threads: int)"
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

def _as_existing_filepaths(value: Any) -> tuple[list[str], str]:
    """
    Normalize a tool return value into:
      - a list of existing file paths ("revived files")
      - leftover text to send as a message

    Supported shapes:
      - "/path/to/file.png"
      - ["a.png", "b.txt"]
      - {"files": [...], "text": "..."} (or "message"/"output")
      - JSON string with the dict-shape above
      - multiline string where some lines are file paths
    """

    def resolve_one(p: str) -> list[str]:
        p = (p or "").strip().strip('"').strip("'")
        if not p:
            return []
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            if path.is_file():
                return [str(path)]
            if path.is_dir():
                return [str(x) for x in path.iterdir() if x.is_file()]
        except OSError:
            return []
        return []

    if value is None:
        return ([], "")

    if isinstance(value, dict):
        files_raw = value.get("files") or value.get("file_paths") or value.get("paths") or value.get("file") or []
        if isinstance(files_raw, str):
            files_raw = [files_raw]
        text = value.get("text") or value.get("message") or value.get("output") or ""
        files: list[str] = []
        if isinstance(files_raw, list):
            for item in files_raw:
                if isinstance(item, str):
                    files.extend(resolve_one(item))
        return (files, str(text).strip())

    if isinstance(value, (list, tuple, set)):
        files: list[str] = []
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                resolved = resolve_one(item)
                if resolved:
                    files.extend(resolved)
                else:
                    texts.append(item)
            else:
                texts.append(str(item))
        return (files, "\n".join(t for t in texts if t.strip()).strip())

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                return _as_existing_filepaths(parsed)
            except Exception:
                pass

        files: list[str] = []
        non_file_lines: list[str] = []
        for line in s.splitlines():
            resolved = resolve_one(line)
            if resolved:
                files.extend(resolved)
            else:
                non_file_lines.append(line)
        return (files, "\n".join(non_file_lines).strip())

    return ([], str(value).strip())


async def _send_text_chunks(channel: discord.abc.Messageable, text: str, limit: int = 2000) -> None:
    text = (text or "").strip()
    if not text:
        return
    for i in range(0, len(text), limit):
        await channel.send(text[i : i + limit])


# === TXT folder pre-load on startup ===
logs, docs = load_txts_multithreaded(TXT_FOLDER, DEFAULT_THREADS)
print("[Blitzkreig TXT Load Results]")
for logline in logs:
    print(logline)
print(f"Total loaded documents: {len(docs)}\n")

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
                file_paths, msg_text = _as_existing_filepaths(out)
                for p in file_paths[:10]:
                    await message.channel.send(file=File(p))
                if len(file_paths) > 10:
                    await _send_text_chunks(
                        message.channel,
                        f"(Sent 10 files; {len(file_paths) - 10} more not sent:)\n" + "\n".join(file_paths[10:]),
                    )
                await _send_text_chunks(message.channel, msg_text)
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
            raw_lower = raw.lower()
            prefixes = (f"!blitzkreig {name}", f"!bk {name}")
            prefix = next((p for p in prefixes if raw_lower == p or raw_lower.startswith(p + " ")), None)
            if prefix:
                args = raw[len(prefix) :].strip()
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
