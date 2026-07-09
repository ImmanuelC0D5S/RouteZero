# RouteZero — File-by-File Function Specification

Maps every file in your repo tree to the exact classes/functions it should contain, based on the locked architecture (Semantic Cache → Semantic+Heuristic Router → Local Qwen / Remote Fireworks → Verifier → Finalize) and AMD Hackathon Act II Track 1 rules (containerized, standardized scoring env, routing-intelligence-first).

**Locked decisions:**
- ❌ No RouteLLM — MF needs external API, causal-LLM too slow
- ❌ No Llama Guard — safety classifier, cannot make routing decisions
- ❌ No Llama 8B — Qwen 3.x MoE activates ~3B params vs Llama's full 8B
- ✅ Semantic + Heuristic Router — prototypes in RAM, zero extra DB, zero extra model
- ✅ Qwen 3.x MoE (30B-A3B) local, Fireworks AI remote
- ✅ ChromaDB cache, all-MiniLM-L6-v2 embedder (shared between cache + router)

---

## `routezero/config.py`

Central source of truth. Everything else imports from here — no magic numbers scattered in other files.

```python
class Settings(BaseSettings):        # pydantic-settings, reads .env
    # Cache
    cache_similarity_threshold: float = 0.92
    chromadb_path: str = ".chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Router — Semantic + Heuristic (no external API needed)
    router_local_threshold: float = 0.4   # complexity score below this → LOCAL
    router_heuristic_weight: float = 0.3  # blend weight for heuristic signal

    # Local model
    local_model_name: str            # Qwen 3.x MoE weights path/HF id
    local_model_endpoint: str        # vLLM/ROCm OpenAI-compatible server URL
    local_max_tokens: int = 1024
    local_timeout_s: float = 30.0

    # Remote model
    fireworks_api_key: str
    fireworks_model_id: str
    remote_timeout_s: float = 60.0

    # Verification
    verification_task_types: list[str] = ["json", "code", "reasoning"]

    # Telemetry
    sqlite_path: str = "metrics.db"

def get_settings() -> Settings: ...   # cached singleton (lru_cache)
```

**Also holds:**
- `RouteTarget(Enum): LOCAL, REMOTE`
- `CacheStatus(Enum): HIT, MISS`
- `TaskType(Enum): FACTUAL, MATH, SENTIMENT, SUMMARIZE, NER, DEBUG, REASONING, CODEGEN`
  *(maps 1:1 to the 8 hackathon evaluation categories)*

---

## `routezero/cache.py`

Owns the semantic memory gate — the first screening gate in the pipeline.

```python
class SemanticCache:
    def __init__(self, settings: Settings): ...
    def _get_collection(self) -> chromadb.Collection: ...

    def embed(self, text: str) -> list[float]: ...
        # sentence-transformers all-MiniLM-L6-v2, 384-dim
        # NOTE: this embedder instance is also passed to the router at startup
        # to avoid loading the model twice

    def query(self, prompt: str) -> CacheHitResult | None:
        # KNN search, cosine similarity, threshold gate (>=0.92)
        # returns cached response + similarity score, or None

    def insert(self, prompt: str, response: str, metadata: dict) -> None:
        # write-through after any verified response (local or remote)

    def stats(self) -> CacheStats:
        # total entries, hit ratio (for dashboard)

@dataclass
class CacheHitResult:
    response: str
    similarity: float
    task_type: str

@dataclass
class CacheStats:
    total_entries: int
    hit_ratio: float
```

---

## `routezero/router.py`

Semantic + Heuristic router — **no external API, no extra model**. Uses the same embedder as the cache (passed in at init). Prototypes live in RAM as numpy arrays.

```python
# Category prototype examples — pre-computed at startup into 384-dim centroids
CATEGORY_EXAMPLES: dict[TaskType, list[str]] = {
    TaskType.FACTUAL:   ["explain how X works", "what is a transformer model", "define recursion"],
    TaskType.MATH:      ["calculate 15% of 340", "project revenue at 8% growth", "solve for x"],
    TaskType.SENTIMENT: ["classify the sentiment", "is this review positive or negative"],
    TaskType.SUMMARIZE: ["summarize in one sentence", "condense to 50 words", "tldr this passage"],
    TaskType.NER:       ["extract all person and location entities", "identify organizations mentioned"],
    TaskType.DEBUG:     ["find the bug in this code", "why does this function return None"],
    TaskType.REASONING: ["if A>B and B>C then", "all conditions must be satisfied", "deduce from"],
    TaskType.CODEGEN:   ["write a function that", "implement a class that does", "create a script"],
}

# Complexity tier — which task types are hard enough to warrant REMOTE by default
REMOTE_TASK_TYPES: set[TaskType] = {TaskType.REASONING, TaskType.CODEGEN, TaskType.DEBUG}
LOCAL_TASK_TYPES:  set[TaskType] = {TaskType.FACTUAL, TaskType.MATH, TaskType.SENTIMENT,
                                     TaskType.SUMMARIZE, TaskType.NER}

class SemanticHeuristicRouter:
    def __init__(self, embedder, settings: Settings):
        # embedder: the same SentenceTransformer instance from SemanticCache
        self.prototypes: dict[TaskType, np.ndarray] = {}

    def _build_prototypes(self) -> None:
        # at startup: embed each category's examples, np.mean(axis=0) → centroid
        # stored as self.prototypes[TaskType] = np.ndarray(384,)

    def classify_task(self, prompt: str) -> TaskType:
        # embed prompt → cosine similarity against all 8 centroids → argmax

    def heuristic_signals(self, prompt: str) -> float:
        # returns 0.0–1.0 complexity boost based on:
        #   - prompt length (longer = more complex)
        #   - presence of code blocks (``` or indentation)
        #   - numeric density (many numbers → math, but simple math)
        #   - constraint phrases ("must satisfy", "all conditions", "step by step")
        #   - explicit output format demands ("return JSON", "write a function")

    def decide(self, prompt: str) -> RoutingDecision:
        # 1. classify_task → get TaskType
        # 2. base_score = 1.0 if task in REMOTE_TASK_TYPES else 0.0
        # 3. heuristic_boost = heuristic_signals(prompt) * settings.router_heuristic_weight
        # 4. final_score = base_score + heuristic_boost
        # 5. if final_score < settings.router_local_threshold → LOCAL else REMOTE
        # returns RoutingDecision

@dataclass
class RoutingDecision:
    target: RouteTarget
    task_type: TaskType
    confidence: float
    reason: str          # e.g. "codegen task + constraint phrases detected → REMOTE"
```

---

## `routezero/llm_clients.py`

Both model backends behind one interface so `pipeline.py` never branches on provider-specific code.

```python
class LLMClient(Protocol):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...

class LocalQwenClient(LLMClient):
    def __init__(self, settings: Settings): ...
    async def generate(self, prompt: str) -> LLMResponse:
        # calls vLLM OpenAI-compatible endpoint (ROCm, AMD Developer Cloud)
        # model: Qwen 3.x MoE 30B-A3B (~3B active params per forward pass)
    async def health_check(self) -> bool: ...

class FireworksRemoteClient(LLMClient):
    def __init__(self, settings: Settings): ...
    async def generate(self, prompt: str) -> LLMResponse:
        # Fireworks AI API call, exponential backoff on 429/5xx
    async def health_check(self) -> bool: ...

@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str
```

---

## `routezero/verifier.py`

Deterministic gate — no LLM self-grading, only programmatic checks. One check per task type, zero retry loops.

```python
def detect_task_type(prompt: str) -> TaskType: ...
    # lightweight regex/heuristic — separate from router's semantic classify
    # used here specifically to pick the right verifier, not for routing

def verify_json(response: str) -> VerificationResult:
    # json.loads() + optional schema key presence check

def verify_code(response: str) -> VerificationResult:
    # extract code block → ast.parse() for Python
    # subprocess compile check for other languages if needed

def verify_reasoning(response: str) -> VerificationResult:
    # regex for intermediate step markers ("Step", "Therefore", "Because")
    # ensures the model didn't just output a bare answer with no chain

def verify_generic(response: str) -> VerificationResult:
    # min length check, not empty, not a refusal string
    # covers FACTUAL, MATH, SENTIMENT, SUMMARIZE, NER

def verify(response: str, task_type: TaskType) -> VerificationResult:
    # dispatcher → calls the right verify_* above

@dataclass
class VerificationResult:
    passed: bool
    failure_reason: str | None
```

---

## `routezero/conversation.py`

Session/context state — keeps the pipeline stateless per-call while supporting multi-turn.

```python
class PipelineState(TypedDict):
    user_prompt: str
    prompt_embedding: list[float]
    cache_hit: bool
    routing_target: str          # "local" | "remote"
    task_type: str
    model_response: str
    verification_passed: bool
    execution_latency_ms: float
    conversation_id: str
    turn_index: int

class ConversationStore:
    def __init__(self): ...
    def get_history(self, conversation_id: str) -> list[dict]: ...
    def append_turn(self, conversation_id: str, role: str, content: str) -> None: ...
    def build_contextual_prompt(self, conversation_id: str, new_prompt: str) -> str:
        # stitches recent turns into a bounded context window
    def new_session(self) -> str:
        # returns a fresh conversation_id (uuid4)
```

---

## `routezero/metrics.py`

Async telemetry — must never block the request path.

```python
class MetricsLogger:
    def __init__(self, settings: Settings): ...
    async def init_db(self) -> None:
        # CREATE TABLE IF NOT EXISTS requests(...)

    async def log_request(self, record: RequestLogRecord) -> None:
        # fire-and-forget async insert via asyncio.create_task

    async def get_dashboard_snapshot(self) -> DashboardSnapshot:
        # cache_hit_ratio, local_offload_rate, remote_fallback_rate,
        # total_token_cost_savings, avg latencies per path,
        # verification_success_rate, routing_breakdown_by_task_type

@dataclass
class RequestLogRecord:
    request_id: str
    cache_status: str        # HIT | MISS
    selected_route: str      # LOCAL | REMOTE | FALLBACK
    task_type: str
    saved_tokens: int
    latency_ms: float
    verification_passed: bool | None
    timestamp: datetime
```

---

## `routezero/pipeline.py`

LangGraph execution graph — wires everything together. Matches Figure 6.1 exactly.

```python
def build_graph(settings: Settings) -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("check_semantic_cache", node_check_cache)
    graph.add_node("route_decision",       node_route_decision)
    graph.add_node("call_local_qwen",      node_call_local)
    graph.add_node("call_remote_api",      node_call_remote)
    graph.add_node("verify_output",        node_verify)
    graph.add_node("finalize_response",    node_finalize)

    graph.set_entry_point("check_semantic_cache")
    graph.add_conditional_edges("check_semantic_cache", route_on_cache_hit,
        {"hit": "finalize_response", "miss": "route_decision"})
    graph.add_conditional_edges("route_decision", route_on_target,
        {"local": "call_local_qwen", "remote": "call_remote_api"})
    graph.add_edge("call_local_qwen", "verify_output")
    graph.add_conditional_edges("verify_output", route_on_verification,
        {"pass": "finalize_response", "fail": "call_remote_api"})
    graph.add_edge("call_remote_api", "finalize_response")
    graph.set_finish_point("finalize_response")
    return graph.compile()

# Node functions (each receives PipelineState, returns updated PipelineState)
async def node_check_cache(state)    -> PipelineState: ...
async def node_route_decision(state) -> PipelineState: ...
async def node_call_local(state)     -> PipelineState: ...
async def node_call_remote(state)    -> PipelineState: ...
async def node_verify(state)         -> PipelineState: ...
async def node_finalize(state)       -> PipelineState: ...

# Conditional-edge predicates
def route_on_cache_hit(state)    -> str: ...  # "hit" | "miss"
def route_on_target(state)       -> str: ...  # "local" | "remote"
def route_on_verification(state) -> str: ...  # "pass" | "fail"

class RouteZeroPipeline:
    def __init__(self, settings: Settings):
        # init cache, router (pass embedder from cache), llm_clients, verifier, metrics
    async def run(self, prompt: str, conversation_id: str | None = None) -> PipelineResult: ...
```

---

## `routezero/__init__.py`

```python
from .config import Settings, get_settings, RouteTarget, TaskType
from .pipeline import RouteZeroPipeline
from .cache import SemanticCache
from .router import SemanticHeuristicRouter
from .llm_clients import LocalQwenClient, FireworksRemoteClient
from .verifier import verify, detect_task_type
from .metrics import MetricsLogger

__all__ = [...]
__version__ = "0.1.0"
```

---

## `app.py` (repo root)

FastAPI entrypoint — reads `/input/tasks.json`, writes `/output/results.json` (hackathon scoring requirement).

```python
app = FastAPI(title="RouteZero")

@app.on_event("startup")
async def startup():
    # init pipeline (which inits cache, builds router prototypes, warms up clients)
    # if /input/tasks.json exists → run batch mode immediately

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse: ...

@app.get("/health")
async def health() -> dict: ...

@app.get("/metrics/dashboard")
async def dashboard() -> DashboardSnapshot: ...

# CRITICAL for hackathon scoring:
async def run_batch_from_file(input_path: str, output_path: str) -> None:
    # reads [{"task_id": "t1", "prompt": "..."}, ...]
    # runs pipeline.run() for each
    # writes [{"task_id": "t1", "answer": "..."}, ...] to output_path
```

---

## `benchmark.py` (repo root)

Proof-of-performance harness — generates cost/speed/quality numbers for the pitch.

```python
def load_task_suite(path: str) -> list[BenchmarkTask]: ...
def run_baseline(task: BenchmarkTask) -> BenchmarkResult:    # remote-only (no routing)
def run_routezero(task: BenchmarkTask) -> BenchmarkResult:   # full pipeline
def compute_summary(results: list[BenchmarkResult]) -> BenchmarkSummary:
    # cost_saved_pct, speed_multiplier, quality_retained_pct
    # routing_breakdown: how many went LOCAL vs REMOTE vs CACHE HIT
def main(): ...   # python benchmark.py --suite evaluation/tasks.json
```

---

## Supporting directories

| Path | Contents |
|---|---|
| `evaluation/` | `tasks.json` with sample tasks across all 8 categories, ground-truth answers |
| `scripts/` | `setup_chromadb.py`, `seed_cache.py`, `download_qwen_weights.sh` |
| `docs/` | AGENT.md, architecture diagrams |
| `ui/` | Optional dashboard frontend (reads `/metrics/dashboard`) |
| `Dockerfile` | Multi-stage ROCm base → runtime; entrypoint runs batch mode then keeps server alive |
| `docker-compose.yml` | `routezero-app` service; mounts `/input` and `/output` volumes |

---

## Build order

1. `config.py` → 2. `llm_clients.py` → 3. `cache.py` → 4. `router.py` → 5. `verifier.py` → 6. `conversation.py` → 7. `metrics.py` → 8. `pipeline.py` → 9. `app.py` → 10. `benchmark.py`

Each file only depends on files above it — no circular imports.
