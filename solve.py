from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
CACHE = ROOT / "cache" / "embeddings"
OUT = ROOT / "answer.csv"

# --- фиксированные гиперпараметры финального сабмита ---
TOP_K = 200
RRF_K = 60
DESC_PREFIX = 400
SIM_QUERIES = 10
EMB_MODEL = "cointegrated/rubert-tiny2"
EMB_MAX_CHARS = 256
EMB_BATCH = 64

WEIGHTS = {
    "title": 1.0,
    "full": 1.5,
    "char": 0.5,
    "memory": 1.5,
    "expansion": 0.5,
    "dense": 0.5,
    "microcat": 1.0,
}
LOC_BOOST = 1.3
MICRO_BOOST = 1.1

_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Текст
# ---------------------------------------------------------------------------


def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower().replace("ё", "е")
    s = _WORD_RE.sub(" ", s)
    return " ".join(s.split())


def build_lemma_cache(words, morph) -> dict[str, str]:
    cache: dict[str, str] = {}
    for w in words:
        if w in cache:
            continue
        parsed = morph.parse(w)
        cache[w] = parsed[0].normal_form if parsed else w
    return cache


def tokenize_lemmatized(text: str, lemma_cache: dict[str, str]) -> list[str]:
    return [lemma_cache.get(tok, tok) for tok in normalize_text(text).split()]


def make_full_text(title: str, params: str, desc: str) -> str:
    d = (desc or "")[:DESC_PREFIX]
    return " ".join(x for x in (title or "", params or "", d) if x)


def make_emb_text(title: str, params: str = "") -> str:
    t = normalize_text(title or "")
    p = normalize_text(params or "")
    s = f"{t} {p}".strip() if p else t
    return (s[:EMB_MAX_CHARS] if len(s) > EMB_MAX_CHARS else s) or "__empty__"


def parse_search_filters(text: str) -> dict:
    sn = normalize_text(text or "")
    out = {"min_rating": None, "vid": "", "tip": ""}
    m = re.search(r"рейтинг пользователя (\d) звезд", sn)
    if m:
        out["min_rating"] = float(m.group(1))
    m = re.search(r"вид услуги (.+?)(?= тип услуги| онлайн| рейтинг|$)", sn)
    if m:
        out["vid"] = m.group(1).strip()
    m = re.search(r"тип услуги (.+?)(?= вид услуги| онлайн| рейтинг|$)", sn)
    if m:
        out["tip"] = m.group(1).strip()
    return out


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def load_data():
    train = pd.read_parquet(DATA / "train.parquet")
    bq = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bi = pd.read_parquet(DATA / "benchmark_items.parquet")

    train["item_id"] = train["item_id"].astype(str)
    bi["item_id"] = bi["item_id"].astype(str)
    bq["query_id"] = bq["query_id"].astype(str)

    for col in ("search_location_id", "search_category", "search_is_delivery_search"):
        train[col] = train[col].astype(str)
        bq[col] = bq[col].astype(str)
    for col in ("item_location_id", "item_microcat_id", "item_category_id"):
        train[col] = train[col].astype(str)
        if col in bi.columns:
            bi[col] = bi[col].astype(str)

    train["search_infm_params_text"] = train["search_infm_params_text"].fillna("").astype(str)
    bq["search_infm_params_text"] = bq["search_infm_params_text"].fillna("").astype(str)
    for col in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        train[col] = train[col].fillna("").astype(str)
        bi[col] = bi[col].fillna("").astype(str)

    print(f"train={len(train):,}  queries={len(bq):,}  items={len(bi):,}")
    return train, bq, bi


# ---------------------------------------------------------------------------
# Токенизация / BM25
# ---------------------------------------------------------------------------


def prepare_tokens(texts, morph, lemma_cache: dict):
    norms = [normalize_text(t) for t in tqdm(texts, desc="normalize")]
    words = {w for n in norms for w in n.split()}
    miss = [w for w in words if w not in lemma_cache]
    if miss:
        lemma_cache.update(build_lemma_cache(tqdm(miss, desc="lemma"), morph))
    tokens = [[lemma_cache[w] for w in n.split()] if n else ["__empty__"] for n in norms]
    return tokens, lemma_cache


def tokens_for_queries(queries, lemma_cache, morph):
    miss = set()
    for q in queries:
        for w in normalize_text(q).split():
            if w not in lemma_cache:
                miss.add(w)
    if miss:
        lemma_cache.update(build_lemma_cache(miss, morph))
    return [tokenize_lemmatized(q, lemma_cache) or ["__empty__"] for q in queries]


def build_bm25(doc_tokens):
    retriever = bm25s.BM25(method="lucene")
    retriever.index(doc_tokens, show_progress=True)
    return retriever


def retrieve_bm25(retriever, query_tokens, item_ids, k):
    results = retriever.retrieve(query_tokens, corpus=item_ids, k=k, show_progress=True)
    return [[str(x) for x in row] for row in results.documents]


# ---------------------------------------------------------------------------
# Каналы
# ---------------------------------------------------------------------------


def channel_char_tfidf(doc_texts, query_texts, item_ids, k, batch_size=200):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80000, dtype=np.float32
    )
    X = vec.fit_transform(doc_texts)
    Q = vec.transform(query_texts)
    n_q = Q.shape[0]
    preds = [None] * n_q
    for start in tqdm(range(0, n_q, batch_size), desc="char-tfidf"):
        end = min(start + batch_size, n_q)
        scores = (Q[start:end] @ X.T).toarray()
        if k >= scores.shape[1]:
            idx = np.argsort(-scores, axis=1)
        else:
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(end - start)[:, None]
            order = np.argsort(-scores[row, part], axis=1)
            idx = part[row, order]
        for i, row_idx in enumerate(idx):
            preds[start + i] = [item_ids[j] for j in row_idx]
    return preds


def build_memory_index(train, corpus_ids: set[str]):
    tp = train.copy()
    tp["q_norm"] = tp["search_query"].map(normalize_text)
    tp = tp[tp["item_id"].isin(corpus_ids)]

    mem: dict[str, Counter] = defaultdict(Counter)
    for qn, iid in zip(tp["q_norm"], tp["item_id"]):
        if qn:
            mem[qn][iid] += 1

    micro: dict[str, set[str]] = {}
    for qn, g in tp.groupby("q_norm"):
        if not qn:
            continue
        top = g["item_microcat_id"].value_counts().head(3).index.tolist()
        micro[qn] = {str(x) for x in top}

    expansion: dict[str, list[str]] = defaultdict(list)
    for iid, qn in zip(tp["item_id"], tp["q_norm"]):
        if qn:
            expansion[iid].append(qn)
    expansion = {iid: list(dict.fromkeys(qs)) for iid, qs in expansion.items()}
    return mem, micro, expansion


def channel_memory(query_texts, mem, corpus_ids, k):
    q_norms = [normalize_text(q) for q in query_texts]
    sim_q_list = list(mem.keys())
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, dtype=np.float32
    )
    Xq = vec.fit_transform(sim_q_list) if sim_q_list else None

    preds = []
    for qn in tqdm(q_norms, desc="memory"):
        scores: Counter = Counter()
        if qn in mem:
            scores.update(mem[qn])
        if Xq is not None and qn:
            sims = (vec.transform([qn]) @ Xq.T).toarray().ravel()
            top_idx = (
                np.argpartition(-sims, SIM_QUERIES)[:SIM_QUERIES]
                if len(sims) > SIM_QUERIES
                else np.arange(len(sims))
            )
            for j in top_idx:
                sim = float(sims[j])
                if sim < 0.35:
                    continue
                other = sim_q_list[j]
                if other == qn:
                    continue
                for iid, cnt in mem[other].items():
                    scores[iid] += sim * cnt
        ranked = [iid for iid, _ in scores.most_common(k * 2) if iid in corpus_ids][:k]
        preds.append(ranked)
    return preds


def channel_doc_expansion(corpus, expansion, query_texts, morph, lemma_cache, k):
    texts = []
    for iid, title in zip(corpus["item_id"], corpus["item_title_raw"]):
        extra = " ".join(expansion.get(iid, [])[:20])
        texts.append(f"{title} {extra}".strip())
    doc_tokens, lemma_cache = prepare_tokens(texts, morph, lemma_cache)
    retriever = build_bm25(doc_tokens)
    q_tokens = tokens_for_queries(query_texts, lemma_cache, morph)
    return retrieve_bm25(retriever, q_tokens, corpus["item_id"].tolist(), k), lemma_cache


def _emb_cache_key(texts, tag: str) -> str:
    h = hashlib.md5()
    h.update(EMB_MODEL.encode())
    h.update(tag.encode())
    h.update(str(len(texts)).encode())
    sample = "\n".join(texts[:50] + texts[len(texts) // 2 : len(texts) // 2 + 50] + texts[-50:])
    h.update(sample.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def get_doc_embeddings(texts, model, tag: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}_{_emb_cache_key(texts, tag)}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            print(f"embeddings cache: {path.name}")
            return arr
    print(f"encoding {len(texts):,} docs ({EMB_MODEL})...")
    arr = model.encode(
        texts,
        batch_size=EMB_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(arr, dtype=np.float32)
    np.save(path, arr)
    return arr


def channel_dense(doc_emb, query_texts, item_ids, model, k, batch_size=64):
    q_emb = model.encode(
        [make_emb_text(q, "") for q in query_texts],
        batch_size=EMB_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    q_emb = np.asarray(q_emb, dtype=np.float32)
    n_q = q_emb.shape[0]
    preds = [None] * n_q
    for start in tqdm(range(0, n_q, batch_size), desc="dense"):
        end = min(start + batch_size, n_q)
        scores = q_emb[start:end] @ doc_emb.T
        if k >= scores.shape[1]:
            idx = np.argsort(-scores, axis=1)
        else:
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(end - start)[:, None]
            order = np.argsort(-scores[row, part], axis=1)
            idx = part[row, order]
        for i, row_idx in enumerate(idx):
            preds[start + i] = [item_ids[j] for j in row_idx]
    return preds


def build_microcat_loc_index(corpus, train, corpus_ids):
    pop = Counter(train.loc[train["item_id"].isin(corpus_ids), "item_id"])
    by_ml: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_m: dict[str, list[str]] = defaultdict(list)
    for iid, m, loc in zip(
        corpus["item_id"].astype(str),
        corpus["item_microcat_id"].astype(str),
        corpus["item_location_id"].astype(str),
    ):
        by_ml[(m, loc)].append(iid)
        by_m[m].append(iid)

    def _sort(ids):
        return sorted(ids, key=lambda x: -pop.get(x, 0))

    by_ml = {k: _sort(v) for k, v in by_ml.items()}
    by_m = {k: _sort(v)[:500] for k, v in by_m.items()}
    return by_ml, by_m


def channel_microcat(query_qnorms, query_locs, micro_map, by_ml, by_m, k):
    preds = []
    for qn, loc in tqdm(zip(query_qnorms, query_locs), total=len(query_qnorms), desc="microcat"):
        micros = list(micro_map.get(qn, set()))
        ranked, seen = [], set()
        for m in micros:
            for iid in by_ml.get((m, loc), []):
                if iid not in seen:
                    ranked.append(iid)
                    seen.add(iid)
                if len(ranked) >= k:
                    break
            if len(ranked) >= k:
                break
        if len(ranked) < k:
            for m in micros:
                for iid in by_m.get(m, []):
                    if iid not in seen:
                        ranked.append(iid)
                        seen.add(iid)
                    if len(ranked) >= k:
                        break
                if len(ranked) >= k:
                    break
        preds.append(ranked[:k])
    return preds


def enrich_micro_map_for_rare(query_qnorms, micro_map, mem, sim_threshold=0.45):
    freq = {q: sum(c.values()) for q, c in mem.items()}
    frequent = [q for q, f in freq.items() if f >= 5]
    if len(frequent) < 100:
        frequent = sorted(freq, key=lambda q: -freq[q])[:5000]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, dtype=np.float32
    )
    X = vec.fit_transform(frequent)
    enriched = {k: set(v) for k, v in micro_map.items()}
    rare = [qn for qn in set(query_qnorms) if qn and qn not in micro_map]
    if not rare:
        return enriched
    Q = vec.transform(rare)
    for start in range(0, len(rare), 200):
        end = min(start + 200, len(rare))
        sims = (Q[start:end] @ X.T).toarray()
        for row, qn in zip(sims, rare[start:end]):
            j = int(np.argmax(row))
            if float(row[j]) < sim_threshold:
                continue
            donor = frequent[j]
            if donor in micro_map:
                enriched[qn] = set(micro_map[donor])
    return enriched


# ---------------------------------------------------------------------------
# Слияние
# ---------------------------------------------------------------------------


def rrf_fuse(channel_preds: dict[str, list[list[str]]], weights: dict[str, float]):
    n = len(next(iter(channel_preds.values())))
    preds_out, scores_out = [], []
    for i in range(n):
        scores: dict[str, float] = defaultdict(float)
        for name, preds in channel_preds.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            for rank, iid in enumerate(preds[i]):
                scores[iid] += w / (RRF_K + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        preds_out.append([iid for iid, _ in ranked[:TOP_K]])
        scores_out.append(dict(ranked))
    return preds_out, scores_out


def apply_soft_boosts(preds, scores_lists, query_locs, query_qnorms, item_loc, item_micro, micro_map):
    out = []
    for i, ranking in enumerate(preds):
        base = dict(scores_lists[i])
        qloc, qn = query_locs[i], query_qnorms[i]
        allowed = micro_map.get(qn, set())
        boosted = {}
        for iid, sc in base.items():
            m = 1.0
            if item_loc.get(iid) == qloc:
                m *= LOC_BOOST
            if item_micro.get(iid) in allowed:
                m *= MICRO_BOOST
            boosted[iid] = sc * m
        ranked = sorted(boosted.items(), key=lambda x: -x[1])
        out.append([iid for iid, _ in ranked[:TOP_K]])
    return out


def apply_geo_twostage(preds, query_locs, item_loc):
    out = []
    for ranking, qloc in zip(preds, query_locs):
        same = [iid for iid in ranking if item_loc.get(iid) == qloc]
        other = [iid for iid in ranking if item_loc.get(iid) != qloc]
        out.append((same + other)[:TOP_K])
    return out


def apply_filter_boosts(preds, query_filters, item_rating, item_params):
    out = []
    for i, ranking in enumerate(preds):
        flt = query_filters[i]
        base = {iid: 1.0 / (RRF_K + rank + 1) for rank, iid in enumerate(ranking)}
        boosted = {}
        for iid, sc in base.items():
            m = 1.0
            if flt["min_rating"] is not None:
                ir = item_rating.get(iid)
                if ir is not None and not (isinstance(ir, float) and np.isnan(ir)):
                    m *= 1.25 if ir >= flt["min_rating"] else 0.85
            ip = normalize_text(item_params.get(iid, ""))
            if flt["vid"] and flt["vid"] in ip:
                m *= 1.35
            if flt["tip"] and flt["tip"] in ip:
                m *= 1.25
            boosted[iid] = sc * m
        ranked = sorted(boosted.items(), key=lambda x: -x[1])
        out.append([iid for iid, _ in ranked[:TOP_K]])
    return out


def pad_to_50(primary, *fallbacks):
    seen, out = set(), []
    for lst in (primary, *fallbacks):
        for iid in lst:
            if iid not in seen:
                out.append(iid)
                seen.add(iid)
            if len(out) >= 50:
                return out
    return out


def validate_answer(path: Path, bq, bi):
    ans = pd.read_csv(path, dtype=str)
    valid = set(bi["item_id"])
    assert list(ans.columns) == ["query_id", "answer"]
    assert set(ans["query_id"]) == set(bq["query_id"])
    for _, row in ans.iterrows():
        items = row["answer"].split()
        assert 1 <= len(items) <= 50
        assert len(items) == len(set(items))
        for it in items:
            assert len(it) == 16 and it in valid
    print(f"answer OK: {len(ans)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t0 = time.time()
    train, bq, bi = load_data()
    morph = pymorphy3.MorphAnalyzer()

    bi_ids = bi["item_id"].tolist()
    bi_set = set(bi_ids)
    bi_loc = dict(zip(bi["item_id"], bi["item_location_id"].astype(str)))
    bi_micro = dict(zip(bi["item_id"], bi["item_microcat_id"].astype(str)))
    bi_rating = dict(zip(bi["item_id"], pd.to_numeric(bi["item_rating"], errors="coerce")))
    bi_params = dict(zip(bi["item_id"], bi["item_infm_params_text"].astype(str)))

    q_texts = bq["search_query"].tolist()
    q_locs = bq["search_location_id"].astype(str).tolist()
    q_norms = [normalize_text(q) for q in q_texts]
    q_filters = [parse_search_filters(x) for x in bq["search_infm_params_text"].tolist()]

    mem, micro_map, expansion = build_memory_index(train, bi_set)
    micro_map = enrich_micro_map_for_rare(q_norms, micro_map, mem)

    lemma_cache: dict[str, str] = {}

    print("\n[1/7] BM25 title")
    tok, lemma_cache = prepare_tokens(bi["item_title_raw"].tolist(), morph, lemma_cache)
    title_preds = retrieve_bm25(
        build_bm25(tok), tokens_for_queries(q_texts, lemma_cache, morph), bi_ids, TOP_K
    )

    print("\n[2/7] BM25 full")
    full_texts = [
        make_full_text(t, p, d)
        for t, p, d in zip(bi["item_title_raw"], bi["item_infm_params_text"], bi["item_description_raw"])
    ]
    tok, lemma_cache = prepare_tokens(full_texts, morph, lemma_cache)
    full_preds = retrieve_bm25(
        build_bm25(tok), tokens_for_queries(q_texts, lemma_cache, morph), bi_ids, TOP_K
    )

    print("\n[3/7] char TF-IDF")
    char_preds = channel_char_tfidf(
        [normalize_text(t) for t in bi["item_title_raw"]],
        [normalize_text(q) for q in q_texts],
        bi_ids,
        TOP_K,
    )

    print("\n[4/7] memory")
    mem_preds = channel_memory(q_texts, mem, bi_set, TOP_K)

    print("\n[5/7] doc expansion")
    exp_preds, lemma_cache = channel_doc_expansion(
        bi, expansion, q_texts, morph, lemma_cache, TOP_K
    )

    print("\n[6/7] dense rubert-tiny2")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL)
    doc_emb = get_doc_embeddings(
        [make_emb_text(t, p) for t, p in zip(bi["item_title_raw"], bi["item_infm_params_text"])],
        model,
        tag="bench_title_params",
    )
    dense_preds = channel_dense(doc_emb, q_texts, bi_ids, model, TOP_K)

    print("\n[7/7] microcat channel")
    by_ml, by_m = build_microcat_loc_index(bi, train, bi_set)
    micro_preds = channel_microcat(q_norms, q_locs, micro_map, by_ml, by_m, TOP_K)

    channels = {
        "title": title_preds,
        "full": full_preds,
        "char": char_preds,
        "memory": mem_preds,
        "expansion": exp_preds,
        "dense": dense_preds,
        "microcat": micro_preds,
    }

    print("\nRRF + boosts")
    fused, scores = rrf_fuse(channels, WEIGHTS)
    fused = apply_soft_boosts(fused, scores, q_locs, q_norms, bi_loc, bi_micro, micro_map)
    fused = apply_geo_twostage(fused, q_locs, bi_loc)
    fused = apply_filter_boosts(fused, q_filters, bi_rating, bi_params)

    answers = [
        pad_to_50(fused[i][:50], title_preds[i], full_preds[i], dense_preds[i])
        for i in range(len(bq))
    ]
    answer = pd.DataFrame(
        {"query_id": bq["query_id"].astype(str), "answer": [" ".join(a) for a in answers]}
    )
    answer.to_csv(OUT, index=False)
    validate_answer(OUT, bq, bi)
    print(f"saved {OUT}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
