
# Libraries

import os
import re
import json
import time
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from duckduckgo_search import DDGS
from youtube_transcript_api import YouTubeTranscriptApi


# Configuration - Keys placeholders
OPENAI_API_KEY = ${{DS-Env.OPENAI_API_KEY}}
YOUTUBE_API_KEY = ${{DS-Env.YOUTUBE_API_KEY}}

INDEX_PATH = "chef_faiss_index"

# apply keys to env so LangChain/OpenAI clients pick them up
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Utilities
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Ingestion pipelines
def youtube_search_video_ids(query: str, max_results: int = 5) -> List[str]:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "id",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]

def fetch_youtube_transcript(video_id: str) -> str:
    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([p["text"] for p in parts])
        return clean_text(text)
    except Exception:
        return ""

def duckduckgo_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    # No API key required
    results = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=num):
            results.append({
                "title": r.get("title"),
                "link": r.get("href"),
                "snippet": r.get("body")
            })
    return results

def fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "RAG-Chef/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            t.extract()
        texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"])]
        return clean_text(" ".join([t for t in texts if t]))
    except Exception:
        return ""

# Indexing
class RAGChefAgent:
    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings()
        self.index = None  # FAISS instance
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")  # change model if needed
        self.qa_chain = None

    def _docs_to_index(self, docs: List[Document]) -> None:
        if not docs:
            return
        if self.index:
            self.index.add_documents(docs)
        else:
            self.index = FAISS.from_documents(docs, self.embeddings)
        # persist
        self.index.save_local(self.index_path)

    def ingest_youtube(self, query: str, max_results: int = 5):
        ids = youtube_search_video_ids(query, max_results=max_results)
        docs = []
        for vid in ids:
            txt = fetch_youtube_transcript(vid)
            if not txt:
                continue
            for chunk in self.splitter.split_text(txt):
                docs.append(Document(page_content=chunk, metadata={"source": f"youtube:{vid}", "query": query}))
        self._docs_to_index(docs)

    def ingest_google(self, query: str, max_results: int = 10):
        results = duckduckgo_search(query, num=max_results)
        docs = []
        for res in results:
            link = res.get("link") or res.get("url")
            text = fetch_page_text(link) if link else ""
            if not text:
                continue
            for chunk in self.splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"source": link, "title": res.get("title")}))
        self._docs_to_index(docs)

    def ingest_blogs(self, urls: List[str]):
        docs = []
        for url in urls:
            text = fetch_page_text(url)
            if not text:
                continue
            for chunk in self.splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"source": url}))
        self._docs_to_index(docs)

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = FAISS.load_local(self.index_path, self.embeddings)
        else:
            self.index = None
        if self.index:
            retriever = self.index.as_retriever(search_kwargs={"k": 5})
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a helpful cooking chef assistant. Use the provided context (web, videos, blogs) to answer "
                    "the user's question in a friendly, actionable way. Provide steps, timings, and ingredient notes when relevant.\n\n"
                    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                ),
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt}
            )

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.index:
            raise RuntimeError("Index not loaded. Call load_index() or ingest data first.")
        res = self.qa_chain({"query": question})
        answer = res.get("result")
        docs = res.get("source_documents", [])
        sources = [{"source": d.metadata.get("source"), "excerpt": d.page_content[:300]} for d in docs]
        return {"answer": answer, "sources": sources}

# FastAPI deployment
app = FastAPI()
agent = RAGChefAgent()
# try to load existing index on startup
try:
    agent.load_index()
except Exception:
    pass

class IngestRequest(BaseModel):
    mode: str  # "youtube" | "google" | "blogs"
    query: str = None
    urls: List[str] = None
    max_results: int = 5

class AskRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        if req.mode == "youtube":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for youtube")
            agent.ingest_youtube(req.query, max_results=req.max_results or 5)
        elif req.mode == "google":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for google")
            agent.ingest_google(req.query, max_results=req.max_results or 10)
        elif req.mode == "blogs":
            if not req.urls:
                raise HTTPException(status_code=400, detail="urls required for blogs")
            agent.ingest_blogs(req.urls)
        else:
            raise HTTPException(status_code=400, detail="invalid mode")
        agent.load_index()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: AskRequest):
    try:
        out = agent.ask(req.question)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# FastAPI deployment
app = FastAPI()
agent = RAGChefAgent()

# try to load existing index on startup
try:
    agent.load_index()
except Exception:
    pass

class IngestRequest(BaseModel):
    mode: str  # "youtube" | "google" | "blogs"
    query: str = None
    urls: List[str] = None
    max_results: int = 5

class AskRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        if req.mode == "youtube":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for youtube")
            agent.ingest_youtube(req.query, max_results=req.max_results or 5)
        elif req.mode == "google":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for google")
            agent.ingest_google(req.query, max_results=req.max_results or 10)
        elif req.mode == "blogs":
            if not req.urls:
                raise HTTPException(status_code=400, detail="urls required for blogs")
            agent.ingest_blogs(req.urls)
        else:
            raise HTTPException(status_code=400, detail="invalid mode")
        agent.load_index()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: AskRequest):
    try:
        out = agent.ask(req.question)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"index_loaded": agent.index is not None}
