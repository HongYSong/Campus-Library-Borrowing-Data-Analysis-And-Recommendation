from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import math
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


class CampusHybridRecommender:
    def __init__(self, hybrid_weight=0.6, similar_user_k=80, cf_popularity_fallback=0.05):
        self.hybrid_weight = hybrid_weight
        self.similar_user_k = similar_user_k
        self.cf_popularity_fallback = cf_popularity_fallback
        self.data_dir = Path(__file__).resolve().parents[1] / "data"
        self.features_dir = self.data_dir / "features"
        self.processed_dir = self.data_dir / "processed"
        self.load_data()
        self.build_cf_model()
        self.load_best_model_config()

    def load_data(self):
        print("正在加载特征矩阵...")
        self.tfidf_matrix = sp.load_npz(self.features_dir / "book_tfidf_matrix.npz")
        self.books_info = pd.read_pickle(self.features_dir / "books_info.pkl")
        self.ui_weights = pd.read_pickle(self.features_dir / "user_item_weights.pkl")

        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.books_info["BOOK_ID"])}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        self.unique_users = self.ui_weights["USERID"].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.unique_users)}

        self.unique_titles = self.books_info["TITLE"].drop_duplicates().tolist()
        self.title_to_idx = {title: idx for idx, title in enumerate(self.unique_titles)}
        self.book_idx_to_title_idx = np.array([self.title_to_idx[title] for title in self.books_info["TITLE"]])

        # 为交互权重补充 TITLE_IDX
        book_id_to_title = dict(zip(self.books_info["BOOK_ID"], self.books_info["TITLE"]))
        self.ui_weights["TITLE_IDX"] = self.ui_weights["BOOK_ID"].map(
            lambda x: self.title_to_idx.get(book_id_to_title.get(x, ""), 0)
        )

        print("正在加载图书元数据与累计热度...")
        self.title_to_abstract = {}
        self.title_to_total_count = {}
        try:
            # 兼容不同处理产物文件名
            candidates = [
                self.processed_dir / "LENDHIST2019_2020_cleaned.csv",
                self.processed_dir / "borrows_2017_2020.csv",
            ]
            raw_path = next((p for p in candidates if p.exists()), None)
            if raw_path is None:
                raise FileNotFoundError("未找到借阅明细 CSV")
            raw_df = pd.read_csv(raw_path, low_memory=False)
            raw_df["ABSTRACT"] = raw_df.get("ABSTRACT", "").fillna("暂无内容简介。")
            if "TITLE" in raw_df.columns:
                self.title_to_abstract = dict(zip(raw_df["TITLE"], raw_df["ABSTRACT"]))
                self.title_to_total_count = raw_df.groupby("TITLE").size().to_dict()
        except Exception as e:
            print(f"元数据加载失败: {e}")

    def build_cf_model(self):
        row_indices = [self.user_id_to_idx[u] for u in self.ui_weights["USERID"]]
        col_indices = self.ui_weights["TITLE_IDX"].tolist()
        data = self.ui_weights["INTEREST_WEIGHT"]
        self.ui_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.unique_users), len(self.unique_titles)),
        )

        # 当前应用在线推断使用固定 100 维（上限保护）
        n_components = min(100, max(1, min(self.ui_matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.ui_matrix)
        self.item_factors = self.svd.components_.T
        self.title_popularity_scores = self._normalize_scores(np.asarray(self.ui_matrix.sum(axis=0)).ravel())

    def load_best_model_config(self):
        # 默认配置
        self.best_model_name = "Hybrid-SVD-CF(alpha=0.6)"
        self.best_model_type = "hybrid_svd"
        self.best_alpha = 0.6

        result_path = self.features_dir / "model_comparison_results.csv"
        try:
            result_df = pd.read_csv(result_path, encoding="utf-8-sig")
            f1_cols = [c for c in result_df.columns if c.startswith("F1@")]
            if "模型" not in result_df.columns or not f1_cols:
                print("测评结果缺少关键列，使用默认模型")
                return

            best_idx = result_df[f1_cols[0]].astype(float).idxmax()
            model_name = str(result_df.loc[best_idx, "模型"])
            alpha_match = re.search(r"alpha=([0-9.]+)", model_name)
            alpha = float(alpha_match.group(1)) if alpha_match else 0.6

            if model_name.startswith("Hybrid-User-CF"):
                self.best_model_name = model_name
                self.best_model_type = "hybrid_user"
                self.best_alpha = alpha
            elif model_name.startswith("Hybrid-SVD-CF"):
                self.best_model_name = model_name
                self.best_model_type = "hybrid_svd"
                self.best_alpha = alpha
            elif model_name.startswith("Content-Based"):
                self.best_model_name = model_name
                self.best_model_type = "content"
            elif model_name.startswith("User-CF"):
                self.best_model_name = model_name
                self.best_model_type = "user_cf"
            elif model_name.startswith("SVD-CF"):
                self.best_model_name = model_name
                self.best_model_type = "svd_cf"
            elif model_name.startswith("Popularity"):
                self.best_model_name = model_name
                self.best_model_type = "popularity"
            else:
                # BPR / LightGBM 当前 app 未实现在线推断，回退到可用的强基线
                self.best_model_name = f"{model_name} -> fallback: Hybrid-User-CF(alpha=0.2)"
                self.best_model_type = "hybrid_user"
                self.best_alpha = 0.2

            print(f"当前推荐模型: {self.best_model_name}")
        except Exception as e:
            print(f"读取测评结果失败，使用默认模型: {e}")

    @staticmethod
    def _normalize_scores(scores):
        scores = np.asarray(scores, dtype=float).ravel()
        if scores.size == 0:
            return scores
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.zeros_like(scores)
        finite_scores = scores[finite_mask]
        result = np.zeros_like(scores)
        if finite_scores.max() > finite_scores.min():
            result[finite_mask] = (finite_scores - finite_scores.min()) / (finite_scores.max() - finite_scores.min())
        else:
            result[finite_mask] = finite_scores
        return result

    def get_user_history(self, target_user_id, top_n=10):
        if target_user_id not in self.user_id_to_idx:
            return []
        user_history = self.ui_weights[self.ui_weights["USERID"] == target_user_id]
        user_history = user_history.sort_values(by="INTEREST_WEIGHT", ascending=False)
        if top_n is not None:
            user_history = user_history.head(top_n)
        history_list = []
        for rank, row in enumerate(user_history.itertuples()):
            book_id = row.BOOK_ID
            if book_id in self.book_id_to_idx:
                idx = self.book_id_to_idx[book_id]
                title = self.books_info.iloc[idx]["TITLE"]
                history_list.append(
                    {
                        "rank": rank + 1,
                        "book_id": book_id,
                        "title": title,
                        "weight": round(row.INTEREST_WEIGHT, 4),
                    }
                )
        return history_list

    def get_content_based_scores(self, target_user_id):
        user_history = self.ui_weights[self.ui_weights["USERID"] == target_user_id]
        if user_history.empty:
            return np.zeros(len(self.books_info))
        top_history = user_history.sort_values(by="INTEREST_WEIGHT", ascending=False).head(3)
        book_indices = [self.book_id_to_idx[b] for b in top_history["BOOK_ID"] if b in self.book_id_to_idx]
        if not book_indices:
            return np.zeros(len(self.books_info))
        user_profile_vector = np.asarray(self.tfidf_matrix[book_indices].mean(axis=0))
        return self._normalize_scores(cosine_similarity(user_profile_vector, self.tfidf_matrix).flatten())

    def get_svd_cf_scores(self, target_user_id):
        if target_user_id not in self.user_id_to_idx:
            return np.zeros(len(self.books_info))
        u_idx = self.user_id_to_idx[target_user_id]
        cf_title_scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        cf_scores = cf_title_scores[self.book_idx_to_title_idx]
        return self._normalize_scores(cf_scores)

    def get_user_cf_scores(self, target_user_id):
        if target_user_id not in self.user_id_to_idx:
            return self.title_popularity_scores[self.book_idx_to_title_idx]

        u_idx = self.user_id_to_idx[target_user_id]
        sims = cosine_similarity(self.ui_matrix[u_idx], self.ui_matrix).ravel()
        sims[u_idx] = 0.0
        positive_idx = np.flatnonzero(sims > 0)

        if len(positive_idx) == 0:
            title_scores = self.title_popularity_scores.copy()
        else:
            if len(positive_idx) > self.similar_user_k:
                top_users = positive_idx[np.argpartition(sims[positive_idx], -self.similar_user_k)[-self.similar_user_k :]]
            else:
                top_users = positive_idx
            top_sims = sims[top_users]
            raw_title_scores = np.asarray(top_sims @ self.ui_matrix[top_users]).ravel()
            title_scores = self._normalize_scores(raw_title_scores)
            title_scores = self._normalize_scores(
                (1 - self.cf_popularity_fallback) * title_scores
                + self.cf_popularity_fallback * self.title_popularity_scores
            )
        return title_scores[self.book_idx_to_title_idx]

    def get_popularity_scores(self):
        return self.title_popularity_scores[self.book_idx_to_title_idx]

    def get_best_model_scores(self, target_user_id):
        cb_scores = self.get_content_based_scores(target_user_id)
        svd_scores = self.get_svd_cf_scores(target_user_id)
        user_cf_scores = self.get_user_cf_scores(target_user_id)
        popularity_scores = self.get_popularity_scores()

        if self.best_model_type == "hybrid_user":
            return self._normalize_scores(self.best_alpha * cb_scores + (1 - self.best_alpha) * user_cf_scores)
        if self.best_model_type == "hybrid_svd":
            return self._normalize_scores(self.best_alpha * cb_scores + (1 - self.best_alpha) * svd_scores)
        if self.best_model_type == "content":
            return cb_scores
        if self.best_model_type == "user_cf":
            return user_cf_scores
        if self.best_model_type == "svd_cf":
            return svd_scores
        if self.best_model_type == "popularity":
            return popularity_scores
        return self._normalize_scores(self.best_alpha * cb_scores + (1 - self.best_alpha) * user_cf_scores)

    def recommend(self, target_user_id, top_n=10):
        if target_user_id not in self.user_id_to_idx:
            return []

        final_scores = self.get_best_model_scores(target_user_id)

        user_history_ids = self.ui_weights[self.ui_weights["USERID"] == target_user_id]["BOOK_ID"].tolist()
        history_indices = [self.book_id_to_idx[b] for b in user_history_ids if b in self.book_id_to_idx]
        history_titles = set(self.books_info.iloc[history_indices]["TITLE"].tolist())

        recommended_indices = []
        seen_titles = set(history_titles)
        for idx in np.argsort(final_scores)[::-1]:
            title = self.books_info.iloc[idx]["TITLE"]
            if title not in seen_titles:
                recommended_indices.append(idx)
                seen_titles.add(title)
            if len(recommended_indices) == top_n:
                break

        recommendations = []
        for rank, idx in enumerate(recommended_indices):
            title = self.books_info.iloc[idx]["TITLE"]
            abstract = self.title_to_abstract.get(title, "暂无内容简介。")
            if len(str(abstract)) > 80:
                abstract = str(abstract)[:80] + "..."
            total_count = self.title_to_total_count.get(title, 0)
            recommendations.append(
                {
                    "rank": rank + 1,
                    "book_id": self.idx_to_book_id[idx],
                    "title": title,
                    "score": round(float(final_scores[idx]), 4),
                    "abstract": abstract,
                    "total_count": int(total_count),
                }
            )
        return recommendations


def paginate_user_history(user_id, page, page_size=10):
    full_history = system.get_user_history(user_id, top_n=None)
    global_max_weight = max((float(x.get("weight", 0)) for x in full_history), default=0.0)
    history_total = len(full_history)
    history_total_pages = max(1, math.ceil(history_total / page_size)) if history_total > 0 else 0
    page = min(max(1, int(page)), history_total_pages) if history_total_pages > 0 else 1
    start = (page - 1) * page_size
    end = start + page_size
    page_history = full_history[start:end]
    return page_history, history_total, history_total_pages, page, global_max_weight


def enrich_history_items(history_items, global_max_weight=0.0):
    enriched = []
    for item in history_items:
        title = item.get("title", "")
        entry = dict(item)
        entry["abstract"] = str(system.title_to_abstract.get(title, "暂无内容简介。"))
        entry["total_count"] = int(system.title_to_total_count.get(title, 0))
        if global_max_weight > 0:
            entry["weight_ratio"] = float(entry.get("weight", 0)) / global_max_weight * 100.0
        else:
            entry["weight_ratio"] = 0.0
        enriched.append(entry)
    return enriched


print("\n--- 启动智能图书馆后端服务 ---")
system = CampusHybridRecommender(hybrid_weight=0.6)
print("系统加载完毕！")


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    user_history = None
    error_msg = None
    search_id = ""
    history_page = 1
    history_page_size = 10
    history_total = 0
    history_total_pages = 0

    if request.method == "POST":
        search_id = request.form.get("userid", "").strip()
        try:
            history_page = max(1, int(request.form.get("history_page", "1").strip()))
        except ValueError:
            history_page = 1
        if search_id:
            if search_id in system.user_id_to_idx:
                user_history, history_total, history_total_pages, history_page, global_max_weight = paginate_user_history(
                    search_id, history_page, history_page_size
                )
                user_history = enrich_history_items(user_history, global_max_weight)
                recommendations = system.recommend(search_id, top_n=10)
            else:
                error_msg = "未找到该读者的借阅历史，无法生成推荐。"
        else:
            error_msg = "请输入有效的读者 ID。"

    return render_template(
        "index.html",
        user_history=user_history,
        recommendations=recommendations,
        error_msg=error_msg,
        search_id=search_id,
        history_page=history_page,
        history_page_size=history_page_size,
        history_total=history_total,
        history_total_pages=history_total_pages,
        model_name=system.best_model_name,
        title_abstract_map=system.title_to_abstract,
        title_total_map=system.title_to_total_count,
    )


@app.route("/history_page", methods=["POST"])
def history_page():
    user_id = request.form.get("userid", "").strip()
    try:
        page = int(request.form.get("history_page", "1").strip())
    except ValueError:
        page = 1
    page_size = 10

    if not user_id or user_id not in system.user_id_to_idx:
        return jsonify({"ok": False, "error": "invalid_user"}), 400

    user_history, history_total, history_total_pages, page, global_max_weight = paginate_user_history(user_id, page, page_size)
    return jsonify(
        {
            "ok": True,
            "items": enrich_history_items(user_history, global_max_weight),
            "history_total": history_total,
            "history_total_pages": history_total_pages,
            "history_page": page,
            "history_page_size": page_size,
            "history_global_max_weight": global_max_weight,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
