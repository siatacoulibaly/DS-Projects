
# Libraries

import os
import re
import json
import time
import requests
import uuid
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Cookie, Response
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from duckduckgo_search import DDGS
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.responses import HTMLResponse


# Configuration - Keys placeholders
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

INDEX_PATH = "chef_faiss_index"

# apply keys to env so LangChain/GenAI clients pick them up
os.environ["GOOGLE_API_KEY"] = OPENAI_API_KEY

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
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.index = None
        self.llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-3.5-flash")
        self.qa_chain = None

        # Initialize conversation history with system prompt
        self.chat_history = [
            {"role": "system", "content": "You are a friendly personal chef assistant. Answer the user conversationally, helpfully, and concisely, keeping track of previous context."}
        ]

        self.load_index()

    def _docs_to_index(self, docs: List[Document]) -> None:
        if not docs:
            return
        if self.index:
            self.index.add_documents(docs)
        else:
            self.index = FAISS.from_documents(docs, self.embeddings)
        self.index.save_local(self.index_path)
        self.load_index()

    def ingest_youtube(self, query: str, max_results: int = 3):
        ids = youtube_search_video_ids(query, max_results=max_results)
        docs = []
        for vid in ids:
            txt = fetch_youtube_transcript(vid)
            if not txt:
                continue
            for chunk in self.splitter.split_text(txt):
                docs.append(Document(page_content=chunk, metadata={"source": f"https://youtube.com/watch?v={vid}", "query": query}))
        self._docs_to_index(docs)

    def ingest_google(self, query: str, max_results: int = 5):
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

    def load_index(self):
        if os.path.exists(self.index_path):
            try:
                self.index = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                self.index = None
        else:
            self.index = None

        if self.index:
            retriever = self.index.as_retriever(search_kwargs={"k": 4})
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a friendly and helpful cooking chef assistant. Use the provided context to answer "
                    "the user's question with steps, timings, and ingredient notes when relevant.\n\n"
                    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                ),
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt}
            )

    def ask(self, question: str) -> Dict[str, Any]:
        ql = question.lower()
        # If the user asks for a recipe and the index is empty, auto-fetch from DuckDuckGo/YouTube on the fly!
        recipe_keywords = ["recipe", "how to make", "cook", "prepare", "ingredients", "dish", "meal"]
        is_recipe_request = any(kw in ql for kw in recipe_keywords)

        if is_recipe_request and not self.index:
            # Automatically ingest web content for this specific recipe query
            try:
                self.ingest_google(question, max_results=3)
            except Exception:
                pass

        # If we have an active QA chain and an index, use RAG
        sources = []
        answer = None
        if self.index and self.qa_chain:
            try:
                res = self.qa_chain({"query": question})
                answer = res.get("result")
                docs = res.get("source_documents", [])
                sources = [{"source": d.metadata.get("source"), "excerpt": d.page_content[:200]} for d in docs]
                return {"answer": answer, "sources": sources}
            except Exception:
                pass

         # Append user question to history
        self.chat_history.append({"role": "user", "content": question})

        # Fallback: pure conversational response from the LLM when no index/documents are needed
        if not answer:
            response = self.llm.invoke(self.chat_history)
            answer = response.content

        # Append assistant response to history
        self.chat_history.append({"role": "assistant", "content": str(answer)})

        if len(self.chat_history) > 21: # Keep only the last 20 messages (10 user + 10 assistant) plus the system prompt
            self.chat_history = [self.chat_history[0]] + self.chat_history[-20:]

        # Normalize raw_answer into a clean string if it's a list or structured object
        if isinstance(answer, list):
            text_parts = []
            for part in answer:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(str(part["text"]))
                    elif "content" in part:
                        text_parts.append(str(part["content"]))
                elif hasattr(part, "text"):
                    text_parts.append(str(part.text))
                else:
                    text_parts.append(str(part))
            answer_str = "".join(text_parts)
        elif isinstance(answer, dict):
            answer_str = str(answer.get("text") or answer.get("content") or json.dumps(answer))
        else:
            answer_str = str(answer)

        # Remove any lingering raw dictionary artifacts just in case
        if answer_str.startswith("[{") or answer_str.startswith("{"):
            try:
                parsed_json = json.loads(answer_str.replace("'", '"'))
                if isinstance(parsed_json, list):
                    answer_str = "".join([item.get("text", str(item)) for item in parsed_json])
                elif isinstance(parsed_json, dict):
                    answer_str = parsed_json.get("text", answer_str)
            except Exception:
                pass

        return {"answer": answer_str, "sources": sources}


#Store active user sessions in memory
sessions = {}

def get_user_agent(session_id: str) -> RAGChefAgent:
    if session_id not in sessions:
        # Give each user their own isolated index directory
        user_index_path = f"chef_faiss_index_{session_id}"
        sessions[session_id] = RAGChefAgent(index_path=user_index_path)
    return sessions[session_id]

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
def ingest(req: IngestRequest, session_id: str = Cookie(None), response: Response = None):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
        
    agent = get_user_agent(session_id)

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
def ask(req: AskRequest, session_id: str = Cookie(None), response: Response = None):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)

    try:
        out = agent.ask(req.question)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"index_loaded": agent.index is not None}

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Your Personal Chef Assistant</title>
        <!-- Include Marked.js for clean Markdown rendering -->
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; display: flex; justify-content: center; }
            .chat-container { width: 100%; max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            .chat-box { height: 400px; border: 1px solid #ddd; border-radius: 4px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; background: #fafafa; }
            .message { margin-bottom: 10px; padding: 8px 12px; border-radius: 6px; max-width: 80%; }
            .user-msg { background: #007bff; color: white; margin-left: auto; text-align: right; }
            .bot-msg { background: #e9ecef; color: #333; }
            .input-group { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h2>🍳 Your personal Chef Assistant</h2>
            <div id="chat-box" class="chat-box"></div>
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Ask a cooking question..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            async function sendMessage() {
                const inputField = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const question = inputField.value.trim();
                if (!question) return;

                chatBox.innerHTML += `<div class="message user-msg">${question}</div>`;
                inputField.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    if (!response.ok) {
                        const errData = await response.json().catch(() => ({}));
                        let errorText = `Server status ${response.status}`;
                        if (errData.detail) {
                            errorText = typeof errData.detail === 'string' 
                                ? errData.detail 
                                : JSON.stringify(errData.detail);
                        }
                        throw new Error(errorText);
                    }
                    
                    const data = await response.json();
                    const answer = data.answer || "Sorry, I received an empty response.";
                    // Parse markdown into clean HTML
                    const formattedAnswer = marked.parse(answer);

                    // If video or web sources were retrieved, append them nicely
                    if (data.sources && data.sources.length > 0) {
                        let sourceLinks = '<br><br><small style="color: #666;"><strong>Sources:</strong><ul style="margin: 2px 0; padding-left: 15px;">';
                        data.sources.forEach(src => {
                            if (src.source) {
                                sourceLinks += `<li><a href="${src.source}" target="_blank" style="color: #007bff; text-decoration: none;">${src.source}</a></li>`;
                            }
                        });
                        sourceLinks += '</ul></small>';
                        formattedAnswer += sourceLinks;
                    }

                    chatBox.innerHTML += `<div class="message bot-msg">${formattedAnswer}</div>`;
                } catch (err) {
                    chatBox.innerHTML += `<div class="message bot-msg">Error connecting to server.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
