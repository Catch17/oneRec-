# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import traceback
import numpy as np

from src.inference import RecommenderService

app = FastAPI(title="oneRec API", version="0.1.0")
svc = None

def to_jsonable(obj):
    # numpy 标量 -> Python 标量
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # 容器递归处理
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in obj]

    return obj


@app.on_event("startup")
def startup_event():
    global svc
    try:
        svc = RecommenderService()
        print("RecommenderService loaded")
    except Exception as e:
        print("RecommenderService init failed:", e)
        traceback.print_exc()
        svc = None


# 简单内存会话（演示版）
SESSION_MEMORY: Dict[str, List[str]] = {}


class RecommendRequest(BaseModel):
    user_id: int
    topk: int = 5


class ChatRequest(BaseModel):
    session_id: str
    user_id: int
    message: str
    topk: int = 5


@app.get("/")
def root():
    return {"message": "oneRec API is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "service_ready": svc is not None}


@app.get("/users")
def users(limit: int = 50):
    if svc is None:
        raise HTTPException(status_code=503, detail="service not ready")

    try:
        ids = getattr(svc, "available_user_ids", sorted(svc.user_sequences.keys()))
        return {"count": len(ids), "user_ids": ids[:limit]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"/users failed: {e}")


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if svc is None:
        raise HTTPException(status_code=503, detail="service not ready")

    try:
        print(f"[DEBUG] req={req}")
        recs = svc.recommend_by_user(user_id=req.user_id, topk=req.topk)
        if recs is None:
            recs = []
        recs = to_jsonable(recs)

        print(f"[DEBUG] recs_len={len(recs)}")
        return {
            "user_id": int(req.user_id),
            "topk": int(req.topk),
            "items": recs
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"/recommend failed: {e}")


@app.post("/chat")
def chat(req: ChatRequest):
    if svc is None:
        raise HTTPException(status_code=503, detail="service not ready")

    try:
        SESSION_MEMORY.setdefault(req.session_id, [])
        SESSION_MEMORY[req.session_id].append(req.message)

        recs = svc.recommend_by_user(user_id=req.user_id, topk=req.topk)
        if recs is None:
            recs = []
        recs = to_jsonable(recs)

        if not recs:
            reply = "我暂时还不够了解你的偏好，先看看热门商品吧。"
        else:
            names = [str(r.get("item_name", r.get("item_id", "未知商品"))) for r in recs[:3]]
            reply = f"根据你最近的行为，我推荐你看看：{', '.join(names)}。"

            print("DEBUG reply =", reply)
            print("DEBUG reply repr =", repr(reply))

        return {
            "session_id": str(req.session_id),
            "memory_len": int(len(SESSION_MEMORY[req.session_id])),
            "assistant_reply": reply,
            "recommendations": recs,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"/chat failed: {e}")


