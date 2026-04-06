import pickle
import torch
from src.model import SASRec


class RecommenderService:
    def __init__(self, bundle_path="artifacts/data_bundle.pkl", model_path="artifacts/best_model.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.num_items = bundle["num_items"]
        self.max_len = bundle["max_seq_len"]
        self.idx2item = bundle.get("idx2item", {})

        train_uids = bundle["train_uids"]
        train_seqs = bundle["train_seqs"]
        self.user_sequences = {int(uid): list(seq) for uid, seq in zip(train_uids, train_seqs)}

        # ✅ 参数名与 model.py 对齐
        self.model = SASRec(
            num_items=self.num_items,
            max_seq_len=self.max_len,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            dropout=0.2
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.available_user_ids = sorted(self.user_sequences.keys())
        print(f"num users: {len(self.available_user_ids)}")
        print(f"sample user_ids: {self.available_user_ids[:20]}")

    def _pad_sequence(self, seq):
        seq = seq[-self.max_len:]
        if len(seq) < self.max_len:
            seq = [0] * (self.max_len - len(seq)) + seq
        return seq

    @torch.no_grad()
    def recommend_by_user(self, user_id: int, topk: int = 5):
        if user_id not in self.user_sequences:
            return []

        seq = self.user_sequences[user_id]
        x = torch.tensor([self._pad_sequence(seq)], dtype=torch.long, device=self.device)

        logits = self.model(x)          # [1, num_items+1]
        last_logits = logits.squeeze(0) # [num_items+1]

        # 不推荐 padding id=0
        if last_logits.shape[0] > 0:
            last_logits[0] = -1e9

        # 过滤已看过
        for item_id in set(seq):
            if 0 < item_id < last_logits.shape[0]:
                last_logits[item_id] = -1e9

        k = min(topk, last_logits.shape[0] - 1)
        scores, indices = torch.topk(last_logits, k=k)

        recs = []
        for s, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
            item_id = int(idx)
            recs.append({
                "item_id": item_id,
                "item_name": self.idx2item.get(item_id, f"item_{item_id}"),
                "score": float(s)
            })
        return recs
