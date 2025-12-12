from sentence_transformers import SentenceTransformer, util
import numpy as np
# from langchain.vectorstores import Chroma
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import Embeddings
from typing import List, Union
import os
import re
import shutil

class EmbeddModel:
    def __init__(self, model_name='/home/ubuntu/playground/models/snowflake-arctic-embed-l-v2.0'):
        self.model = SentenceTransformer(model_name, local_files_only=True, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, prompt_name="document", batch_size=1).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# -----------------------------
# Chroma 向量库
# -----------------------------
class VectorStore:
    def __init__(self, embedding: EmbeddModel,
                 db_path: str = '/home/ubuntu/playground/temp/vectordb'):
        self.embedding = embedding
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.vectorstore = None

    # 文本分块
    def chunk_text(self, text: str) -> List[Document]:
        pattern = r'!\[\]\(data:image[^)]+\)'
        cleaned_text = re.sub(pattern, '', text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

        chunks = text_splitter.split_text(cleaned_text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        return docs

    # 构建 Chroma 索引
    def build_index(self, text: str):
        docs = self.chunk_text(text)

        if self.vectorstore is not None:
            self.vectorstore.add_documents(docs)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding,
                persist_directory=self.db_path
            )

        self.vectorstore.persist()
        print(f"✅ Chroma Vector DB built & saved at {self.db_path}")

    # 加载现有 Chroma 索引
    def load_index(self):
        print(f"📦 Loading Chroma index from {self.db_path} ...")
        self.vectorstore = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.db_path
        )
        print("✅ Load success.")

    def reset_db(self):
        # 1️⃣ 删除目录下所有文件
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"🗑 Deleted DB directory: {self.db_path}")

        # 2️⃣ 重新创建空目录
        os.makedirs(self.db_path, exist_ok=True)

        # 3️⃣ 将 vectorstore 设为 None
        self.vectorstore = None
        print("✅ VectorStore reset complete. You can now build a new DB.")


    # 搜索
    def search(self, query: str, top_k: int = 5):
        if self.vectorstore is None:
            return {"error": "向量数据库未构建或未加载，请先 build_index() 或 load_index()"}

        results = self.vectorstore.similarity_search(query, k=top_k)

        output = []
        for doc in results:
            output.append({
                "content": doc.page_content,
            })
        return output


# -----------------------------
# 对外搜索接口
# -----------------------------
def main_embed(query: str, vector_store: VectorStore = None):
    if vector_store is None:
        return 'vector_store not initialized'
    result = vector_store.search(query)
    return result


# if __name__ == "__main__":
#     embed_model = EmbeddModel()
#     vec = VectorStore(embed_model)

#     text = '''
# orms the foundation for effectively optimizing preferences. However, training a model to learn from an individual preference pair for each example may not fully capture the complexity of human learning. Human cognition often involves interpreting divergent responses, not only to identical questions but also to similar ones, highlighting the multifaceted nature of comprehension and preference formation [Dahlin et al., 2018]. Moreover, obtaining pairwise preference data can pose challenges and incur substantial costs, especially in sensitive domains such as healthcare and personal services, where careful attention to ethical considerations is essential [Murtaza et al., 2023].

# Our inspiration draws from the human learning process, where valuable insights often arise from the comparison of successful examples and relevant failures [Dahlin et al., 2018]. To emulate this, we introduce Relative Preference Optimization (RPO). This approach involves analyzing prompt similarities at the semantic level within each mini-batch, allowing us to classify pairs as either highly related or unrelated. We construct a contrast matrix that instructs the model to distinguish between preferred and dispreferred responses, applicable to both identical and semantically related prompts. We have developed three weighting strategies to recalibrate the comparison of each contrastive pair. Our findings reveal that reweighting based on prompt similarities significantly enriches model alignment with human preferences, offering a more nuanced understanding. Furthermore, RPO inherently excels in handling non-pairwise preference data by considering semantically related contrastive pairs.

# As illustrated in Figure 1, we are interested in the question “Explain the concept of photosynthesis.” DPO applies penalties for incorrect responses and rewards for precise responses generated for the specific prompt. Conversely, our method RPO emphasizes the semantic connections between various prompts. For instance, the prompt “Describe the importance of sunlight in plant growth” is conceptually similar, and its responses might intersect with those of the initial question. Under RPO, if an answer is less preferred for the second prompt, it is also treated as less suitable for the first prompt. Thus, RPO penalizes both $y_{l,1}$ and $y_{l,2}$ while approving $y_{w,1}$. It is crucial to note that not all prompts are semantically related enough to form effective contrastive pairs. RPO incorporates a reweighting mechanism, whereby unrelated prompts are given less emphasis during training. RPO expands the learning horizon of the model, empowering it to leverage insights from a broader range of prompts, mirroring the human learning process more closely.

# We empirically evaluate RPO on three LLMs, LLaMA2-7/13B and Mistral-7B. We compare RPO with
# '''
#     vec.build_index(text)
