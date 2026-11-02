"""Pinecone (serverless) vector store with local FastEmbed embeddings.

Every uploaded book / PDF set lives in its own Pinecone *namespace*, chosen by the user.
"""
import time
from functools import lru_cache
from typing import List, Optional

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import get_settings


@lru_cache
def _embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=get_settings().embedding_model)


@lru_cache
def _client() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key)


@lru_cache
def _index():
    """The Pinecone index; created on first use."""
    s = get_settings()
    pc = _client()

    # Dimension must match the embedding model (bge-small-en-v1.5 -> 384),
    # so we measure it instead of hard-coding it.
    if not pc.has_index(s.pinecone_index):
        dim = len(_embeddings().embed_query("dimension probe"))
        pc.create_index(
            name=s.pinecone_index,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
        while not pc.describe_index(s.pinecone_index).status["ready"]:
            time.sleep(1)

    return pc.Index(s.pinecone_index)


@lru_cache(maxsize=None)
def get_vectorstore(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(index=_index(), embedding=_embeddings(), namespace=namespace)


def _existing_index():
    """The index if it exists, else None (never creates it)."""
    s = get_settings()
    return _client().Index(s.pinecone_index) if _client().has_index(s.pinecone_index) else None


def namespaces() -> dict:
    """{namespace: chunk count} for everything stored."""
    idx = _existing_index()
    if idx is None:
        return {}
    stats = idx.describe_index_stats()
    return {name: ns.vector_count for name, ns in stats.namespaces.items()}


def add_texts(texts: List[str], metadatas: Optional[List[dict]] = None,
              namespace: Optional[str] = None) -> int:
    """Chunk raw texts and upsert them into a namespace. Returns number of chunks stored."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(texts, metadatas=metadatas)
    get_vectorstore(namespace or get_settings().pinecone_namespace).add_documents(docs)
    return len(docs)


def search(query: str, k: int, namespace: Optional[str] = None) -> List[Document]:
    """Top-k chunks from one namespace, or from all namespaces when namespace is None."""
    if namespace:
        return get_vectorstore(namespace).similarity_search(query, k=k)
    scored = []
    for ns in namespaces():
        scored.extend(get_vectorstore(ns).similarity_search_with_score(query, k=k))
    scored.sort(key=lambda pair: pair[1], reverse=True)  # cosine: higher = closer
    return [doc for doc, _ in scored[:k]]


def find_duplicate(file_hash: str, filename: str) -> Optional[dict]:
    """Is this PDF already stored? Matches on identical content or on the same file name.

    The file name check also catches PDFs uploaded before content hashes were recorded.
    Pinecone lists new records a few seconds after they are written, so a PDF uploaded
    moments ago may not be found yet.
    """
    probe = _embeddings().embed_query("duplicate check")
    for ns in namespaces():
        store = get_vectorstore(ns)
        for match, flt in (("identical content", {"file_hash": file_hash}),
                           ("same file name", {"source": filename})):
            hits = store.similarity_search_by_vector(probe, k=1, filter=flt)
            if hits:
                return {"namespace": ns, "match": match,
                        "source": hits[0].metadata.get("source", filename),
                        "partial": bool(hits[0].metadata.get("partial"))}
    return None


def stats() -> dict:
    """Vector counts per namespace (handy for checking ingest worked)."""
    idx = _existing_index()
    return idx.describe_index_stats().to_dict() if idx else {"namespaces": {}, "total_vector_count": 0}


def _scan(max_vectors: int = 5000, samples_per_doc: int = 3) -> dict:
    """Page through every namespace and group records by (namespace, 'source' metadata).

    Pinecone has no 'list documents' call, so this lists vector ids and reads their
    metadata. It stops after max_vectors so a huge index can't make it slow.
    """
    counts = namespaces()
    total_chunks = sum(counts.values())
    groups: dict = {}
    idx = _existing_index()
    seen = 0
    truncated = False

    for ns in counts:
        if truncated:
            break
        for ids in idx.list(namespace=ns):
            for i in range(0, len(ids), 100):
                fetched = idx.fetch(ids=ids[i:i + 100], namespace=ns)
                for vec in fetched.vectors.values():
                    meta = vec.metadata or {}
                    key = (ns, meta.get("source") or "(raw text, no file name)")
                    g = groups.setdefault(key, {"chunks": 0, "pages": set(), "samples": []})
                    g["chunks"] += 1
                    if "page" in meta:
                        g["pages"].add(int(meta["page"]))
                    if len(g["samples"]) < samples_per_doc and meta.get("text"):
                        g["samples"].append(meta["text"])
            seen += len(ids)
            if seen >= max_vectors:
                truncated = seen < total_chunks
                break
    return {"namespaces": counts, "total_chunks": total_chunks, "groups": groups, "truncated": truncated}


def documents() -> dict:
    """Chunk totals, per-namespace counts and one row per document."""
    scan = _scan(samples_per_doc=0)
    docs = [{"name": name, "namespace": ns, "chunks": g["chunks"], "pages": len(g["pages"])}
            for (ns, name), g in sorted(scan["groups"].items())]
    return {"total_chunks": scan["total_chunks"], "namespaces": scan["namespaces"],
            "documents": docs, "truncated": scan["truncated"]}


def sample_for_summary(max_docs: int = 15, chars_per_sample: int = 600) -> dict:
    """A few text snippets from each document, for the LLM content summary."""
    scan = _scan(samples_per_doc=3)
    docs = []
    for (ns, name), g in sorted(scan["groups"].items())[:max_docs]:
        docs.append({"name": name, "namespace": ns, "chunks": g["chunks"],
                     "snippets": [t[:chars_per_sample] for t in g["samples"]]})
    return {"total_chunks": scan["total_chunks"], "documents": docs, "truncated": scan["truncated"]}
