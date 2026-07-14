import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from performance_tracker import empty_performance

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB", "neurolearn")

if not MONGO_URI:
    print("[DB] WARNING: MONGO_URI not set. Add it to backend/.env or set it as an environment variable.")
    print("[DB] Example: MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority")

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tlsCAFile=certifi.where(),
    )
    db = client[DB_NAME]
    try:
        await db.sessions.create_index("session_id", unique=True)
        print(f"[DB] Connected to MongoDB ({DB_NAME})")
    except Exception as exc:
        print(f"[DB] Warning: Could not verify MongoDB connection: {exc}")
        print("[DB] The app will retry on first request.")


async def close_db():
    global client
    if client:
        client.close()


def get_collection():
    return db["sessions"]

def get_users_collection():
    return db["users"]

def get_database():
    return db

async def create_session(session_id: str, subject: str, user_id: str = None) -> dict:
    doc = {
        "session_id": session_id,
        "user_id": user_id,
        "subject": subject,
        "level": "unknown",
        "total_correct": 0,
        "total_attempts": 0,
        "level_history": [],
        "performance": empty_performance(),
        "created_at": datetime.now(timezone.utc),
    }
    await get_collection().insert_one(doc)
    return {
        "id": session_id,
        "subject": subject,
        "level": "unknown",
        "total_correct": 0,
        "total_attempts": 0,
        "level_history": [],
        "performance": doc["performance"],
    }


async def get_session(session_id: str) -> dict | None:
    doc = await get_collection().find_one({"session_id": session_id})
    if doc is None:
        return None
    return {
        "id": doc["session_id"],
        "subject": doc["subject"],
        "level": doc["level"],
        "total_correct": doc["total_correct"],
        "total_attempts": doc["total_attempts"],
        "level_history": doc.get("level_history", []),
        "performance": doc.get("performance", empty_performance()),
    }


async def update_session(session_id: str, **fields):
    """Update arbitrary fields on the session document."""
    if not fields:
        return
    await get_collection().update_one(
        {"session_id": session_id},
        {"$set": fields},
    )


async def get_sessions_by_user(user_id: str) -> list[dict]:
    """
    Return all sessions for a given user, ordered newest-first (max 50).
    Only fetches summary fields — not the full performance blob.
    """
    cursor = get_collection().find(
        {"user_id": user_id},
        {
            "session_id": 1,
            "subject": 1,
            "level": 1,
            "total_correct": 1,
            "total_attempts": 1,
            "created_at": 1,
            "performance.mastery_score": 1,
        },
    ).sort("created_at", -1).limit(50)

    docs   = await cursor.to_list(length=50)
    result = []
    for doc in docs:
        created = doc.get("created_at", "")
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        result.append({
            "session_id":     doc["session_id"],
            "subject":        doc.get("subject", ""),
            "level":          doc.get("level", "unknown"),
            "total_correct":  doc.get("total_correct", 0),
            "total_attempts": doc.get("total_attempts", 0),
            "mastery":        doc.get("performance", {}).get("mastery_score", 0.0),
            "created_at":     str(created),
        })
    return result

