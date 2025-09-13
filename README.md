


# Generative-Ai-workflow

# 1. Data Ingestion & Preparation

**Sources:** text docs (PDF, Word), tables (CSV, SQL), images, audio, video, APIs, web  
**Tools/Frameworks:** LangChain, LlamaIndex, Haystack, custom ETL  

**Techniques:**
- Text extraction (PDFPlumber, Apache Tika)
- OCR (Tesseract, LayoutLM, DocTR)
- Speech-to-text (Whisper, Deepgram)
- Video transcription/scene detection

---

# 2. Chunking & Segmentation

- **Text:** fixed-size, sliding window, recursive (LangChain), semantic split  
- **Tables:** row-wise, column-wise, schema-aware chunking  
- **Images:** patch-based, caption-based (BLIP, SAM)  
- **Audio:** transcript-based chunking  
- **Video:** frame sampling, scene segmentation, timeline-based splits  

---

# 3. Embeddings (Vectorization)

Convert raw inputs into dense vectors that capture semantic meaning → used for similarity, retrieval, clustering.

### Inputs & Outputs by Modality:
- **Text:** string → vector (e.g., 768–1536 dims)
- **Tables:** row/column → vector capturing structured meaning
- **Images:** pixels → vector encoding visual features
- **Audio:** waveform/spectrogram → vector encoding phonetic/semantic features
- **Video:** frames + audio → temporal multimodal vector

### Popular Models:
- **Text (general):** OpenAI text-embedding-3, Cohere, HuggingFace E5, MiniLM, Instructor
- **Domain-specific:** BioBERT, SciBERT, FinBERT, LegalBERT
- **Multilingual:** LaBSE, mUSE, multilingual-E5
- **Images:** CLIP, BLIP, Florence
- **Audio:** Wav2Vec2, Whisper embeddings
- **Video:** VideoCLIP, VIOLET

---

# 4. Vector Databases / Vector Stores

- **Open-source:** FAISS, Milvus, Weaviate, Qdrant, pgvector  
- **Managed/Cloud:** Pinecone, Chroma Cloud, Vertex AI Matching Engine, Azure Cognitive Search, AWS Kendra/OpenSearch  
- **Hybrid Search:** Vespa, Elastic, Weaviate (BM25 + dense)  

---

# 5. Indexing & Search Techniques

- **Structures:** Flat, IVF, HNSW, PQ  
- **Hybrid Search:** combine sparse (BM25) + dense (embeddings)  

**Specialized Indexing:**
- Text → inverted + semantic
- Tables → schema/key indexing
- Images → perceptual hashing + vectors
- Audio/Video → fingerprinting, temporal indexing

---

# 6. Retrieval & Augmentation

**Frameworks:** LangChain retrievers, LlamaIndex query engines, Haystack retrievers  

**Techniques:**
- Top-K similarity
- Maximal Marginal Relevance (MMR)
- Reranking: cross-encoders (Cohere Rerank, bge-reranker)
- Adaptive retrieval (context window control)

---

# 7. Generation Layer (LLM Integration)

**LLMs:** GPT-4/5, Claude, Llama 3, Gemini, Mistral, Falcon  
**Frameworks:** LangChain, LlamaIndex, Semantic Kernel  

**Strategies:**
- Direct RAG prompting
- Multi-query retrieval
- Chain-of-thought, citations, tool-augmented responses

---

# 8. Orchestration & Application Layer

**Frameworks:** LangChain, LlamaIndex, Semantic Kernel, Haystack, DSPy  
**Agents & Pipelines:** LangChain Agents, CrewAI, AutoGPT  
**Integrations:** REST APIs, GraphQL, enterprise connectors (SharePoint, Salesforce, Slack)

---

# 9. Evaluation & Monitoring

**Metrics:** precision@k, recall, MRR, nDCG, hallucination rate  
**Tools:** Ragas, DeepEval, TruLens, LangSmith, Arize AI, Weights & Biases  
**Continuous Improvement:** human feedback loops, active learning  

---

# 10. Deployment & Scaling

**Serving:** FastAPI, BentoML, TorchServe, HuggingFace Inference Endpoints  
**Platforms:** Kubernetes, Docker, Ray, Airflow  
**Enterprise Concerns:** auth, security, compliance (GDPR, HIPAA), caching (Redis, Vespa), cost optimization


## 🆓 Free LLM / AI APIs (Groq-like)

| Provider | What you get free | How to sign up |
|----------|------------------|---------------|
| **[Groq Cloud](https://console.groq.com)** | Free tier with **LLaMA-3**, **Mixtral**, **Gemma** models at very high speed (100s tokens/sec). No credit card required. | Go to [console.groq.com](https://console.groq.com) → “Create API Key”. |
| **[Together AI](https://api.together.xyz)** | Free credits for open-weight models (LLaMA-3, Mistral, Mixtral, etc.) – ~100k tokens/day free for development. | Sign up at [api.together.xyz](https://api.together.xyz). |
| **[Replicate](https://replicate.com)** | Many community models (LLMs, vision, audio). Gives free “Starter” credits monthly. | Sign in with GitHub at [replicate.com](https://replicate.com). |
| **[Hugging Face Inference Endpoints](https://api-inference.huggingface.co)** | Free “Hosted Inference API” for most open models (limited rate). Example: `https://api-inference.huggingface.co/models/...` | Get an access token at [huggingface.co](https://huggingface.co) → Settings → Access Tokens. |
| **[OpenRouter](https://openrouter.ai)** | Free trial tokens across multiple providers/models (some open models are always free). | Sign up at [openrouter.ai](https://openrouter.ai). |
| **[Cohere](https://cohere.com)** | Free tier for text generation, classification, embeddings (with limits). | Sign up at [cohere.com](https://cohere.com) and get your API key. |


corrective rag


# --- Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# --- Embeddings + Vectorstore ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

urls = ["url1", "url2", "url3"]
docs = [doc for url in urls for doc in WebBaseLoader(url).load()]
splits = RecursiveCharacterTextSplitter(chunk_size=250).split_documents(docs)
vectorstore = Chroma.from_documents(splits, collection_name="rag-chroma", embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- RAG Chain ---
prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()

# --- Grader ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")
structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade relevance of doc vs question (yes/no)."),
    ("human", "Doc:\n{document}\n\nQuestion: {question}")
])
retrieval_grader = grade_prompt | structured_llm_grader

# --- Question Rewriter ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question for web search optimization."),
    ("human", "Question: {question}")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# --- Web Search ---
web_search_tool = TavilySearchResults(k=3)

# --- Graph Functions ---
def retrieve(state): ...
def grade_documents(state): ...
def generate(state): ...
def transform_query(state): ...
def web_search(state): ...
def decide_to_generate(state): ...

# --- Graph State ---
class State(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: list

# --- Build Workflow ---
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
                               {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- Run ---
inputs = {"question": "tell me about the Taj Mahal"}
for output in app.stream(inputs):
    print(output)
print("Final Answer:", output["generate"]["generation"])

| **Anthropic Claude (via [Poe](https://poe.com) or Slack integrations)** | No public API free plan, but free usage of certain Claude models through Poe.com. API itself is paid. | — |
| **[Vercel AI SDK / Open-Source Models](https://vercel.com/docs/ai)** | Deploy open-weight models to Vercel Edge Functions; free tier covers some requests. | See [vercel.com/docs/ai](https://vercel.com/docs/ai). |
| **[DeepInfra](https://deepinfra.com)** | Pay-as-you-go but always gives a small free quota to new accounts; hosts open models. | Sign up at [deepinfra.com](https://deepinfra.com). |


