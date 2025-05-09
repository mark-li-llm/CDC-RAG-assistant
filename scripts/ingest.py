"""
One-shot ingest script.
Run once to embed all .txt docs and push them into Qdrant (cloud or local).
使用: python scripts/ingest.py [--rebuild]
选项:
  --rebuild    强制重建索引，删除现有集合
"""

import os
import sys
from dotenv import load_dotenv
from index_graph.graph import _load_docs           # 复用读取 + 切块
from qdrant_client import QdrantClient, models as qmodels
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

# 检查是否需要重建集合
force_rebuild = "--rebuild" in sys.argv

# 加载.env文件中的环境变量（使用正确的绝对路径）
def load_env():
    """加载环境变量，使用绝对路径确保正确加载"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    env_path = os.path.join(root_dir, '.env')
    
    print(f"尝试从路径加载环境变量: {env_path}")
    load_dotenv(env_path)
    
    # 调试信息
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        key_preview = f"{openai_api_key[:5]}...{openai_api_key[-4:]}"
        print(f"已从 {env_path} 加载 OPENAI_API_KEY: {key_preview}")
    else:
        print(f"警告：未能从 {env_path} 加载 OPENAI_API_KEY")

# 调用环境变量加载函数
load_env()

# 获取并打印环境变量值进行调试
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
print(f"环境变量QDRANT_URL: {qdrant_url}")
print(f"环境变量QDRANT_API_KEY: {'已设置' if qdrant_api_key else '未设置'}")

# URL格式检查与修正
if qdrant_url:
    # 移除可能的node-0-前缀
    if "node-0-" in qdrant_url:
        qdrant_url = qdrant_url.replace("node-0-", "")
        print(f"已移除node-0-前缀，URL现在是: {qdrant_url}")
    
    # 确保URL包含端口号6333
    if not (":6333" in qdrant_url):
        # 处理URL可能有尾随斜杠的情况
        if qdrant_url.endswith("/"):
            qdrant_url = qdrant_url[:-1] + ":6333"
        else:
            qdrant_url = qdrant_url + ":6333"
        print(f"已添加端口号，URL现在是: {qdrant_url}")

# 将修正后的URL设置回环境变量
os.environ["QDRANT_URL"] = qdrant_url

COLLECTION = "articles_demo"
LOCAL_COLLECTION = "articles_demo_local"  # 本地备份集合名称
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# ── 同时创建云端和本地客户端 ───────────────────────────────────────────────
# 创建本地客户端
local_client = QdrantClient(path="./qdrant_data")
print("✅ 已连接到本地Qdrant")

# 如果配置了云端，则同时连接云端
cloud_client = None
if qdrant_url:
    try:
        cloud_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            check_compatibility=False
        )
        print("✅ 已连接到云端Qdrant")
    except Exception as e:
        print(f"❌ 连接云端Qdrant失败: {e}")
        print("⚠️ 将只使用本地存储")

# ── 处理云端集合 ───────────────────────────────────────────────
# 查看和重建云端集合
if cloud_client:
    try:
        cloud_collections = [c.name for c in cloud_client.get_collections().collections]
        print(f"云端现有集合: {cloud_collections}")
        
        if COLLECTION in cloud_collections and force_rebuild:
            print(f"🔄 删除云端旧集合 '{COLLECTION}'")
            cloud_client.delete_collection(collection_name=COLLECTION)
        elif COLLECTION in cloud_collections:
            print(f"ℹ️ 使用现有云端集合 '{COLLECTION}'")
    except Exception as e:
        print(f"⚠️ 处理云端集合时出错: {e}")
        cloud_client = None  # 出错时禁用云端

# ── 处理本地集合 ───────────────────────────────────────────────
# 查看和重建本地集合
local_collections = [c.name for c in local_client.get_collections().collections]
print(f"本地现有集合: {local_collections}")

if LOCAL_COLLECTION in local_collections and force_rebuild:
    print(f"🔄 删除本地旧集合 '{LOCAL_COLLECTION}'")
    local_client.delete_collection(collection_name=LOCAL_COLLECTION)
elif LOCAL_COLLECTION in local_collections:
    print(f"ℹ️ 使用现有本地集合 '{LOCAL_COLLECTION}'")

# ── 创建嵌入模型 ───────────────────────────────────────────────
print("初始化OpenAI嵌入模型...")
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key: {'已设置' if openai_api_key else '未设置'}")  # 调试信息

embeddings = OpenAIEmbeddings(
    model=EMB_MODEL,
    openai_api_key=openai_api_key  # 显式传递API Key
)
embedding_size = len(embeddings.embed_query("test"))

# ── 创建集合 ───────────────────────────────────────────────
vector_config = qmodels.VectorParams(
    size=embedding_size,
    distance=qmodels.Distance.COSINE,
)

# 创建本地集合
if LOCAL_COLLECTION not in local_collections or force_rebuild:
    print(f"创建本地集合 '{LOCAL_COLLECTION}'...")
    local_client.create_collection(
        collection_name=LOCAL_COLLECTION,
        vectors_config=vector_config
    )

# 创建云端集合(如果可用)
if cloud_client and (COLLECTION not in cloud_collections or force_rebuild):
    try:
        print(f"创建云端集合 '{COLLECTION}'...")
        cloud_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=vector_config
        )
    except Exception as e:
        print(f"❌ 创建云端集合失败: {e}")
        cloud_client = None  # 出错时禁用云端

# ── 初始化向量存储 ───────────────────────────────────────────────
# 初始化本地向量存储
local_vectorstore = QdrantVectorStore(
    client=local_client,
    collection_name=LOCAL_COLLECTION,
    embedding=embeddings,
)

# 初始化云端向量存储(如果可用)
cloud_vectorstore = None
if cloud_client:
    try:
        cloud_vectorstore = QdrantVectorStore(
            client=cloud_client,
            collection_name=COLLECTION,
            embedding=embeddings,
        )
        print("✅ 云端向量存储就绪")
    except Exception as e:
        print(f"❌ 初始化云端向量存储失败: {e}")

# ── 分块文档 ───────────────────────────────────────────────
print("开始分块文档...")
# 使用RecursiveCharacterTextSplitter代替SemanticChunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
docs = splitter.split_documents(_load_docs())
print(f"总计分为 {len(docs)} 个文档块")

# ── 添加文档到向量存储 ───────────────────────────────────────────────
# 判断是否已有数据
local_count = local_client.count(LOCAL_COLLECTION).count
cloud_count = cloud_client.count(COLLECTION).count if cloud_client else 0

# 如果本地集合为空或强制重建，则添加文档
if local_count == 0 or force_rebuild:
    # 调用一次OpenAI API，添加到本地
    print(f"📝 添加到本地: 向量化 {len(docs)} 个文档块...")
    local_vectorstore.add_documents(docs)
    print(f"✅ 本地索引完成: {local_client.count(LOCAL_COLLECTION).count} 条向量")
else:
    print(f"ℹ️ 本地集合已包含 {local_count} 条向量，跳过索引构建（使用 --rebuild 强制重建）")

# 如果云端可用且云端集合为空或强制重建，则添加到云端
if cloud_vectorstore and (cloud_count == 0 or force_rebuild):
    try:
        print(f"📝 添加到云端: 向量化 {len(docs)} 个文档块...")
        cloud_vectorstore.add_documents(docs)
        print(f"✅ 云端索引完成: {cloud_client.count(COLLECTION).count} 条向量")
    except Exception as e:
        print(f"❌ 添加到云端失败: {e}")
        print("⚠️ 数据仅保存在本地")
elif cloud_vectorstore:
    print(f"ℹ️ 云端集合已包含 {cloud_count} 条向量，跳过索引构建（使用 --rebuild 强制重建）")

print("\n✨ 索引构建完成! 数据已备份到本地，随时可从本地恢复，无需重新调用OpenAI API。")
if not force_rebuild and (local_count > 0 or cloud_count > 0):
    print("ℹ️ 使用了现有集合。如需重建索引，请使用 --rebuild 参数。")
