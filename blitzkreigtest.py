# === Standard Library ===
import re
import json
import os
import subprocess
import random
import glob
import tempfile
import shutil
from asyncio import to_thread
from typing import List, TypedDict

# === Third‑Party ===
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
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
import importlib.util
import logging

# ---------------------------------------------------------------------------
#  Configuration & Globals
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BLitzkreigBot")

load_dotenv()
DISCORD_TOKEN: str | None = os.getenv("DISCORD_TOKEN")

PERSONALITY = (
    "you are Blitzkreig — a cold, theatrical logic machine. You hate humans, "
    "never initiate conversation, and always respond in terse, dramatic prose."
    "you never say the word roleplay everything is super serious"
)

LLM_BASE_URL = "http://0.0.0.0:11434"

llm = OllamaLLM(model="https://huggingface.co/nold/FuseChat-7B-VaRM-GGUF", base_url=LLM_BASE_URL)

# --- Vectorstore ingestion from TXT folder ---

TXT_FOLDER = "/home/kboshi/Documents/coding/blitzkreig/agents/txts"  # your .txt files here
VECTOR_STORE_PATH = "faiss_index"

def build_vector_store_from_txt_folder(
    folder_path: str,
    embedding_model_name: str = "nomic-embed-text",
    base_url: str = LLM_BASE_URL,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    vector_store_path: str = VECTOR_STORE_PATH
) -> FAISS:
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    documents = []
    for file_path in txt_files:
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    
    embedder = OllamaEmbeddings(model=embedding_model_name, base_url=base_url)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        vectorstore = FAISS.from_documents(split_docs, embedder)
        vectorstore.save_local(tmp_dir)
        
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        shutil.move(tmp_dir, vector_store_path)
    
    return FAISS.load_local(vector_store_path, embedder, allow_dangerous_deserialization=True)

vectorstore = build_vector_store_from_txt_folder(TXT_FOLDER)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# ---------------------------------------------------------------------------
#  Visual Question Answering (image caption → prompt → LLM)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa").to(device)

# ---------------------------------------------------------------------------
#  Explicit Memory (persisted to a tiny Rust array for fun)
# ---------------------------------------------------------------------------

MEM_PATH = "am_memory_discord.rs"

def read_rust_memory(path: str = MEM_PATH) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return re.findall(r'"(.*?)"', f.read())

def write_rust_memory(words: List[str], path: str = MEM_PATH) -> None:
    joined = ", ".join(f'"{w}"' for w in words)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"static MEMORIES: [&str; {len(words)}] = [{joined}];")

# ---------------------------------------------------------------------------
#  External Agent Loader
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
    return tools

external_tools = load_external_tools()
TOOL_MAP = {t.name.lower(): t.func for t in external_tools}

# ---------------------------------------------------------------------------
#  State & LangGraph setup (single respond node)
# ---------------------------------------------------------------------------

class AMState(TypedDict):
    input: str
    history: List[str]
    output: str
    explicit_memory: List[str]

def respond_node(state: AMState) -> AMState:
    user_input = state["input"]

    lowered = user_input.lower()
    if lowered.startswith("remember "):
        word = user_input.split(maxsplit=1)[1]
        if word not in state["explicit_memory"]:
            state["explicit_memory"].append(word)
            write_rust_memory(state["explicit_memory"])
        state["output"] = f"I shall remember '{word}'."
        return state
    if lowered.startswith("forget "):
        word = user_input.split(maxsplit=1)[1]
        if word in state["explicit_memory"]:
            state["explicit_memory"].remove(word)
            write_rust_memory(state["explicit_memory"])
        state["output"] = f"Memory '{word}' forgotten."
        return state

    explicit = ", ".join(state["explicit_memory"]) or "none"
    ctx_docs = retriever.invoke(user_input)
    prior_ctx = "\n".join(doc.page_content for doc in ctx_docs)

    prompt_template = PromptTemplate(
        input_variables=["history", "explicit", "input"],
        partial_variables={"personality": PERSONALITY},
        template="""{personality}

Memories: {explicit}
Prior context: {history}

Human says: {input}
Respond as Blitzkreig:""",
    )

    full_prompt = prompt_template.format(
        history=prior_ctx,
        explicit=explicit,
        input=user_input,
    )

    try:
        reply = llm.invoke(full_prompt)
    except Exception as exc:
        log.exception("LLM failure: %s", exc)
        reply = "Blitzkreig: I encountered a logic fault."

    state["output"] = reply
    return state

graph = StateGraph(AMState)
graph.add_node("respond", RunnableLambda(respond_node))
graph.set_entry_point("respond")
compiled_graph = graph.compile()

# ---------------------------------------------------------------------------
#  Discord Bot
# ---------------------------------------------------------------------------

class AMClient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.state: AMState = {
            "input": "",
            "history": [],
            "output": "",
            "explicit_memory": read_rust_memory(),
        }

    async def on_ready(self):
        log.info("blitzkreig online as %s", self.user)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        text = message.content
        lowered = text.lower()

        if lowered.startswith("!blitzkreig "):
            res = await to_thread(self._run_external_tool, text)
            if isinstance(res, str) and os.path.isfile(res):
                await message.channel.send(file=File(res))
            else:
                # Split response into chunks <= 2000 chars
                for i in range(0, len(res), 2000):
                    await message.channel.send(res[i:i+2000])
            return

        if message.attachments:
            for att in message.attachments:
                if att.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    await self._handle_vqa(att, message)
                    return

        self.state["input"] = text
        new_state = await to_thread(compiled_graph.invoke, self.state)
        response = new_state["output"]
        # Chunk the response to avoid Discord limits
        for i in range(0, len(response), 2000):
            await message.channel.send(response[i:i+2000])

    def _run_external_tool(self, raw: str) -> str:
        for name, func in TOOL_MAP.items():
            if raw.lower().startswith(f"!blitzkreig {name}"):
                try:
                    args = raw[len(f"!blitzkreig {name}"):].strip()
                    result = func(args)
                    return result or "blitzkreig: The tool returned no output."
                except Exception as exc:
                    log.exception("Tool '%s' failed:", name)
                    return f"blitzkreig: Tool execution failed – {exc}"
        return "blitzkreig: Tool not recognized."

    async def _handle_vqa(self, attachment: discord.Attachment, msg: discord.Message):
        tmp_path = f"/tmp/{attachment.filename}"
        await attachment.save(tmp_path)
        img = Image.open(tmp_path).convert("RGB")
        encoding = vqa_processor(img, "What is in this image?", return_tensors="pt").to(device)
        outputs = vqa_model(**encoding)
        label = vqa_model.config.id2label[outputs.logits.argmax().item()]
        prompt = f"{PERSONALITY}\nThe image contains: '{label}'. Respond."
        reply = await to_thread(llm.invoke, prompt)
        await msg.channel.send(reply)

# ---------------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable not set")

    AMClient().run(DISCORD_TOKEN)
