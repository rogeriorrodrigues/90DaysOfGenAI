import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType, Configure

# Conecta no localhost:8080 (Docker)
client = weaviate.connect_to_local()

# Coleção sem vectorizer interno — vamos enviar os vetores
client.collections.create(
    name="Docs",
    properties=[Property(name="text", data_type=DataType.TEXT)],
    vector_config=Configure.Vectors.self_provided()
)

docs = ["Guia de LGPD", "ISO 27001: controles", "RAG com Weaviate"]
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(docs, normalize_embeddings=True).tolist()

col = client.collections.get("Docs")
for d, v in zip(docs, vecs):
    col.data.insert({"text": d}, vector=v)

q = model.encode("Como cumprir LGPD?", normalize_embeddings=True).tolist()
res = col.query.near_vector(near_vector=q, limit=1)
print(res.objects[0].properties["text"])
client.close()
