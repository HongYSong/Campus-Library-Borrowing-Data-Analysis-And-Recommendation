from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


# ==========================================
# 1. 核心推荐算法类 (统计全校借阅总热度)
# ==========================================
class CampusHybridRecommender:
    def __init__(self, hybrid_weight=0.6):
        self.hybrid_weight = hybrid_weight
        self.load_data()
        self.build_cf_model()

    def load_data(self):
        print("正在加载特征矩阵...")
        self.tfidf_matrix = sp.load_npz('../data/features/book_tfidf_matrix.npz')
        self.books_info = pd.read_pickle('../data/features/books_info.pkl')
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.books_info['BOOK_ID'])}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        self.ui_weights = pd.read_pickle('../data/features/user_item_weights.pkl')
        self.unique_users = self.ui_weights['USERID'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.unique_users)}

        self.unique_titles = self.books_info['TITLE'].drop_duplicates().tolist()
        self.title_to_idx = {title: idx for idx, title in enumerate(self.unique_titles)}
        self.book_idx_to_title_idx = np.array([self.title_to_idx[title] for title in self.books_info['TITLE']])
        
        # 建立临时映射用于快速查 TITLE
        book_id_to_title = dict(zip(self.books_info['BOOK_ID'], self.books_info['TITLE']))
        self.ui_weights['TITLE_IDX'] = self.ui_weights['BOOK_ID'].map(lambda x: self.title_to_idx.get(book_id_to_title.get(x, ""), 0))


        print("正在挖掘图书元数据与全量历史热度...")
        try:
            raw_df = pd.read_csv('../data/processed/LENDHIST2019_2020_cleaned.csv')

            # 1. 提取简介
            raw_df['ABSTRACT'] = raw_df['ABSTRACT'].fillna('暂无内容简介。')
            self.title_to_abstract = dict(zip(raw_df['TITLE'], raw_df['ABSTRACT']))

            # 2. 【核心修改】：按书名统计全校累计借阅次数 (包含所有不同 ID 的副本)
            # 使用 size() 统计每个书名出现的总行数
            self.title_to_total_count = raw_df.groupby('TITLE').size().to_dict()

        except Exception as e:
            print(f"元数据加载失败: {e}")
            self.title_to_abstract = {}
            self.title_to_total_count = {}

    def build_cf_model(self):
        row_indices = [self.user_id_to_idx[u] for u in self.ui_weights['USERID']]
        col_indices = self.ui_weights['TITLE_IDX'].tolist()
        data = self.ui_weights['INTEREST_WEIGHT']
        
        # 基于 TITLE_IDX 构建矩阵，极大克服复本冗余造成的分化
        self.ui_matrix = sp.csr_matrix((data, (row_indices, col_indices)),
                                       shape=(len(self.unique_users), len(self.unique_titles)))
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = self.svd.fit_transform(self.ui_matrix)
        self.item_factors = self.svd.components_.T

    def get_user_history(self, target_user_id, top_n=10):
        if target_user_id not in self.user_id_to_idx:
            return []
        user_history = self.ui_weights[self.ui_weights['USERID'] == target_user_id]
        user_history = user_history.sort_values(by='INTEREST_WEIGHT', ascending=False).head(top_n)

        history_list = []
        for rank, row in enumerate(user_history.itertuples()):
            book_id = row.BOOK_ID
            if book_id in self.book_id_to_idx:
                idx = self.book_id_to_idx[book_id]
                title = self.books_info.iloc[idx]['TITLE']
                history_list.append({
                    'rank': rank + 1,
                    'book_id': book_id,
                    'title': title,
                    'weight': round(row.INTEREST_WEIGHT, 4)
                })
        return history_list

    def get_content_based_scores(self, target_user_id):
        user_history = self.ui_weights[self.ui_weights['USERID'] == target_user_id]
        if user_history.empty:
            return np.zeros(len(self.books_info))
        top_history = user_history.sort_values(by='INTEREST_WEIGHT', ascending=False).head(3)
        book_indices = [self.book_id_to_idx[b] for b in top_history['BOOK_ID']]
        user_profile_vector = np.asarray(self.tfidf_matrix[book_indices].mean(axis=0))
        return cosine_similarity(user_profile_vector, self.tfidf_matrix).flatten()

    def get_collaborative_scores(self, target_user_id):
        if target_user_id not in self.user_id_to_idx:
            return np.zeros(len(self.books_info))
        u_idx = self.user_id_to_idx[target_user_id]
        cf_title_scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        
        # 降维的 Title 得分，映射广播回高维的 BOOK_ID 维度
        cf_scores = cf_title_scores[self.book_idx_to_title_idx]
        
        if cf_scores.max() > cf_scores.min():
            cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
        return cf_scores

    def recommend(self, target_user_id, top_n=10):
        if target_user_id not in self.user_id_to_idx:
            return []

        cb_scores = self.get_content_based_scores(target_user_id)
        cf_scores = self.get_collaborative_scores(target_user_id)
        hybrid_scores = (self.hybrid_weight * cb_scores) + ((1 - self.hybrid_weight) * cf_scores)

        user_history_ids = self.ui_weights[self.ui_weights['USERID'] == target_user_id]['BOOK_ID'].tolist()
        history_indices = [self.book_id_to_idx[b] for b in user_history_ids if b in self.book_id_to_idx]
        history_titles = set(self.books_info.iloc[history_indices]['TITLE'].tolist())

        recommended_indices = []
        seen_titles = set(history_titles)

        for idx in np.argsort(hybrid_scores)[::-1]:
            title = self.books_info.iloc[idx]['TITLE']
            if title not in seen_titles:
                recommended_indices.append(idx)
                seen_titles.add(title)
            if len(recommended_indices) == top_n:
                break

        recommendations = []
        for rank, idx in enumerate(recommended_indices):
            title = self.books_info.iloc[idx]['TITLE']
            abstract = self.title_to_abstract.get(title, "暂无内容简介。")
            if len(str(abstract)) > 80:
                abstract = str(abstract)[:80] + "..."

            # 【数据注入】：获取书名对应的全校累计借阅总数
            total_count = self.title_to_total_count.get(title, 0)

            recommendations.append({
                'rank': rank + 1,
                'book_id': self.idx_to_book_id[idx],
                'title': title,
                'score': round(hybrid_scores[idx], 4),
                'abstract': abstract,
                'total_count': total_count  # 传给前端的变量名改为 total_count
            })
        return recommendations


# ==========================================
# 2. Flask 路由
# ==========================================
print("\n--- 启动智能图书馆后端服务 ---")
system = CampusHybridRecommender(hybrid_weight=0.6)
print("系统加载完毕！")


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    user_history = None
    error_msg = None
    search_id = ""

    if request.method == 'POST':
        search_id = request.form.get('userid', '').strip()
        if search_id:
            if search_id in system.user_id_to_idx:
                user_history = system.get_user_history(search_id, top_n=10)
                recommendations = system.recommend(search_id, top_n=10)
            else:
                error_msg = "未找到该读者的借阅历史，无法生成推荐。"
        else:
            error_msg = "请输入有效的读者 ID。"

    return render_template('index.html',
                           user_history=user_history,
                           recommendations=recommendations,
                           error_msg=error_msg,
                           search_id=search_id)


if __name__ == '__main__':
    app.run(debug=True, port=5000)