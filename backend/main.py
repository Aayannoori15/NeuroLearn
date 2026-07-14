from dotenv import load_dotenv

load_dotenv()  # must be called before any other imports that read env vars

from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from database import connect_db, close_db, get_database
from routes import router, limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    if not os.getenv("MONGODB_URL"):
        raise RuntimeError("MONGODB_URL environment variable is not set")
        
    await connect_db()
    yield
    await close_db()


app = FastAPI(title="NeuroLearn API", version="0.1.0", lifespan=lifespan)

# Rate limiter — must be registered before routes
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "service": "NeuroLearn API"}


@app.get("/health")
async def health():
    """Health check endpoint for deployment monitoring."""
    db = get_database()
    try:
        await db.command("ping")
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "NeuroLearn API",
        "db": db_status
    }
