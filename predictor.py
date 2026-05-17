"""
predictor.py — Unified prediction pipeline for TruthLens

Decision Formula
────────────────
  final_real = 0.60 * avg_sim  +  0.40 * (1 - ann_fake)

Verdict Thresholds
──────────────────
  real ≥ 0.72  →  REAL
  real ≥ 0.58  →  LIKELY REAL
  real ≥ 0.44  →  UNCERTAIN
  real ≥ 0.28  →  LIKELY FAKE
  else         →  FAKE

Special overrides (applied after formula)
──────────────────────────────────────────
  • Hard disinformation phrase detected  → clamp real ≤ 0.15
  • Consensus (2+ Tier-1, sim > 0.72)   → clamp real ≥ 0.93
  • Debunk / fact-check query            → fix real = 0.50 (UNCERTAIN)
  • 'today' query with no fresh articles → always trigger live search

Temporal Intelligence
─────────────────────
  Queries mentioning today/tomorrow/weekdays/months get special treatment:
  • 'today'     → forces live search, freshness-ranked results
  • 'tomorrow'  → fetch forecast/prediction articles (weather, events)
  • week/month  → broader temporal window in search
  • stale cache → automatically triggers Google News fallback
"""

import os
import time
import logging
import threading

import numpy as np
import torch
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from model      import FakeNewsANN
from rag_engine import RAGEngine
from analyzer   import (
    clean_text, get_linguistic_score,
    get_predicates, detect_temporal,
    check_negation, detect_topic, detect_location,
    temporal_label,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
_CONSENSUS_MATCH_TYPE = 'CONSENSUS_VERIFIED'

VERDICT_THRESHOLDS = [
    (0.72, 'REAL'),
    (0.58, 'LIKELY REAL'),
    (0.44, 'UNCERTAIN'),
    (0.28, 'LIKELY FAKE'),
    (0.00, 'FAKE'),
]

VERDICT_META = {
    'REAL':        {'emoji': '✅', 'color': '#22c55e'},
    'LIKELY REAL': {'emoji': '🟢', 'color': '#86efac'},
    'UNCERTAIN':   {'emoji': '⚠️',  'color': '#facc15'},
    'LIKELY FAKE': {'emoji': '🟠', 'color': '#fb923c'},
    'FAKE':        {'emoji': '❌', 'color': '#ef4444'},
}


def _get_verdict(real_score: float) -> str:
    for threshold, label in VERDICT_THRESHOLDS:
        if real_score >= threshold:
            return label
    return 'FAKE'


def _time_ago(ts: float) -> str:
    """Convert a Unix timestamp to a human-readable 'time ago' string.
    
    This is computed from the article's actual publish time so it never
    shows '18 days ago' for recent news — it reflects the real age of the article.
    """
    if not ts:
        return 'Unknown'
    diff = time.time() - ts
    if diff < 0:
        return 'Just published'
    if diff < 60:
        return 'Just now'
    if diff < 3600:
        mins = int(diff / 60)
        return f'{mins}m ago'
    if diff < 86400:
        hrs = int(diff / 3600)
        return f'{hrs}h ago'
    if diff < 172800:
        return 'Yesterday'
    days = int(diff / 86400)
    if days <= 7:
        return f'{days}d ago'
    # If older than a week, show actual date
    from datetime import datetime
    dt = datetime.fromtimestamp(ts)
    return dt.strftime('%d %b %Y')


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class PredictionResult:
    verdict:          str
    confidence:       float   # 0-100
    real_score:       float   # 0-1
    fake_score:       float   # 0-1
    ann_fake_prob:    float   # raw ANN output 0-1
    rag_similarity:   float   # avg cosine 0-1
    # sensationalism:   float   # 0-1
    # attribution:      float   # 0-1
    topic:            str
    location:         str
    temporality:      str
    explanation:      str
    similar_articles: list
    processing_ms:    float

    def to_dict(self) -> dict:
        meta = VERDICT_META.get(self.verdict, {})
        return {
            'verdict':          self.verdict,
            'verdict_emoji':    meta.get('emoji', ''),
            'verdict_color':    meta.get('color', '#888'),
            'confidence':       round(self.confidence,       1),
            'real_score':       round(self.real_score,       4),
            'fake_score':       round(self.fake_score,       4),
            'ann_fake_prob':    round(self.ann_fake_prob,    4),
            'rag_similarity':   round(self.rag_similarity,   4),
            # 'sensationalism':   round(self.sensationalism,   4),
            # 'attribution':      round(self.attribution,      4),
            'topic':            self.topic,
            'location':         self.location,
            'temporality':      self.temporality,
            'explanation':      self.explanation,
            'similar_articles': self.similar_articles,
            'processing_ms':    round(self.processing_ms,   1),
        }


# ── TruthLensPredictor ────────────────────────────────────────

class TruthLensPredictor:
    """
    Combines:
      • ANN writing-pattern detector  (trained on WELFake)
      • RAG semantic retrieval        (live RSS + FAISS)
    into a single calibrated verdict.
    """

    W_RAG = 0.60   # weight of RAG similarity in decision formula
    W_ANN = 0.40   # weight of ANN (1 - ann_fake) in decision formula

    # For 'today'/'tomorrow' queries, always force live search
    FORCE_LIVE_TEMPORALITIES = {'today', 'tomorrow'}

    def __init__(
        self,
        model_path:  str        = 'ann.pth',
        embed_model: str        = 'all-MiniLM-L6-v2',
        device:      str | None = None,
    ):
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        log.info('TruthLensPredictor init on %s …', self._device)

        # 1. Shared sentence encoder (downloaded once, ~80 MB)
        log.info('Loading encoder: %s …', embed_model)
        self._encoder = SentenceTransformer(embed_model, device=self._device)

        # 2. ANN model
        self._ann = FakeNewsANN(input_dim=384).to(self._device)
        self._load_ann(model_path)
        self._ann.eval()

        # 3. RAG engine (shares the same encoder to avoid duplicate GPU/CPU memory)
        self.rag = RAGEngine(self._encoder)

    # ── ANN loading ───────────────────────────────────────────
    def _load_ann(self, path: str) -> None:
        if not os.path.exists(path):
            log.warning(
                "ann.pth not found at '%s'. Using random weights — "
                'run train.py for production accuracy.', path
            )
            return
        try:
            ckpt  = torch.load(path, map_location=self._device)
            state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            self._ann.load_state_dict(state)
            acc = ckpt.get('val_acc', float('nan')) if isinstance(ckpt, dict) else float('nan')
            log.info('ANN weights loaded (val_acc=%.4f)', acc)
        except Exception as exc:
            log.warning('ANN load failed (%s) — using random weights.', exc)

    # ── Predict ───────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, query: str, top_k: int = 8) -> PredictionResult:
        t0    = time.time()
        query = query.strip()

        if len(query) < 4:
            return self._neutral(query, t0, 'Query too short for analysis')

        # ── 1. Linguistic analysis ────────────────────────────
        analysis    = get_linguistic_score(query)
        # sens        = analysis['sensationalism']
        # attr        = analysis['attribution']
        topic       = analysis['topic']
        hard_hit    = analysis['hard_hit']
        temporality = analysis.get('temporality', detect_temporal(query))
        is_debunk   = check_negation(query)
        q_location  = detect_location(query)

        # ── 2. ANN inference ──────────────────────────────────
        q_clean  = clean_text(query)
        q_emb_np = self._encoder.encode(
            [q_clean],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        q_tensor = torch.from_numpy(q_emb_np).float().to(self._device)
        ann_fake = float(self._ann(q_tensor).item())

        # ── 3. Primary RAG retrieval (temporality-aware) ──────
        rag_results, avg_sim = self.rag.search(query, top_k=top_k, temporality=temporality)
        max_sim = max((r.similarity for r in rag_results), default=0.0)

        # ── 4. Live fallback ──────────────────────────────────
        # Always force live search for today/tomorrow queries or weak cache
        live_triggered = False
        force_live = (temporality in self.FORCE_LIVE_TEMPORALITIES) or (max_sim < self.rag.LIVE_FALLBACK)

        if force_live:
            live_hits = self.rag.search_live(query, top_k=8)
            if live_hits:
                combined = rag_results + live_hits
                combined.sort(key=lambda r: r.similarity, reverse=True)
                rag_results    = combined[:top_k]
                max_sim        = max((r.similarity for r in rag_results), default=0.0)
                avg_sim        = float(np.mean([r.similarity for r in rag_results]))
                live_triggered = True

        # ── 5. Consensus detection ────────────────────────────
        consensus = any(r.match_type == _CONSENSUS_MATCH_TYPE for r in rag_results)

        # ── 6. Decision formula ───────────────────────────────
        final_real = self.W_RAG * avg_sim + self.W_ANN * (1.0 - ann_fake)
        final_real = float(np.clip(final_real, 0.0, 1.0))

        # Temporal adjustment: for 'today'/'tomorrow' queries where we have fresh
        # live results, boost confidence slightly (live results are more trustworthy)
        if temporality in ('today', 'tomorrow') and live_triggered and max_sim > 0.45:
            final_real = min(1.0, final_real * 1.08)

        # Hard overrides (applied in order of priority)
        if is_debunk:
            final_real = 0.50                        # fact-check query → UNCERTAIN

        if hard_hit:
            final_real = min(final_real, 0.15)       # known disinfo phrase

        if consensus and avg_sim > 0.72:
            final_real = max(final_real, 0.93)       # cross-source consensus

        final_real = float(np.clip(final_real, 0.0, 1.0))
        final_fake = 1.0 - final_real
        confidence = max(final_real, final_fake) * 100.0

        # ── 7. Build explanation ──────────────────────────────
        reasons: list[str] = []

        # Temporal context label
        temp_label = temporal_label(temporality)
        if temporality not in ('any',):
            reasons.append(f'Query time context: {temp_label}')

        if consensus:
            reasons.append(f'Cross-source consensus: {len(rag_results)} sources agree')
        elif max_sim > 0.80:
            reasons.append(f'High-confidence news match ({max_sim:.0%} similarity)')
        elif live_triggered and temporality in ('today', 'tomorrow'):
            # Count only fresh articles (< 24h) in results
            fresh_count = sum(1 for r in rag_results if r.article.age_hours < 24)
            if fresh_count > 0:
                reasons.append(f'Real-time news: {fresh_count} fresh article(s) within 24h')
            else:
                reasons.append('Live search performed — no articles from past 24h found')
        elif live_triggered:
            reasons.append('Real-time news results used as evidence')
        elif avg_sim > 0.55:
            reasons.append(f'Supported by {len(rag_results)} related news reports')
        else:
            reasons.append('Weak news evidence — pattern-based analysis primary')

        if hard_hit:
            reasons.append('Known disinformation phrase detected')
        # if sens > 0.7:
        #     reasons.append('Highly sensationalist language detected')
        # if attr > 0.5:
            # reasons.append('Attribution language present — increases credibility')
        if is_debunk:
            reasons.append('Query framed as fact-check — returning UNCERTAIN')

        # Freshness warning for today queries with stale top result
        if temporality == 'today' and rag_results:
            oldest_top = max(r.article.age_hours for r in rag_results[:3])
            if oldest_top > 24 and not live_triggered:
                reasons.append('⚠️ No fresh articles found — cache may be behind real-time events')

        explanation = ' | '.join(reasons) if reasons else 'Pattern-based analysis'

        # ── 8. Format similar articles ────────────────────────
        articles_out = [
            {
                'source':     r.article.source,
                'title':      r.article.title,
                'summary':    r.article.summary[:220],
                'url':        r.article.url,
                'similarity': round(r.similarity, 4),
                'match_type': r.match_type,
                'favicon':    r.article.get_favicon(),
                'time_ago':   _time_ago(r.article.timestamp),    # now uses real publish timestamp
                'age_hours':  round(r.article.age_hours, 1),
                'topic':      r.article.topic,
            }
            for r in rag_results
        ]

        ms = (time.time() - t0) * 1000
        log.info(
            "query='%.50s'  topic=%s  loc=%s  temporal=%s  ann=%.3f  sim=%.3f  "
            'real=%.3f  verdict=%s  %.0fms',
            query, topic, q_location, temporality, ann_fake, avg_sim,
            final_real, _get_verdict(final_real), ms,
        )

        return PredictionResult(
            verdict          = _get_verdict(final_real),
            confidence       = confidence,
            real_score       = final_real,
            fake_score       = final_fake,
            ann_fake_prob    = ann_fake,
            rag_similarity   = avg_sim,
            # sensationalism   = sens,
            # attribution      = attr,
            topic            = topic,
            location         = q_location,
            temporality      = temporality,
            explanation      = explanation,
            similar_articles = articles_out,
            processing_ms    = ms,
        )

    def _neutral(self, query: str, t0: float, reason: str = '') -> PredictionResult:
        return PredictionResult(
            verdict='UNCERTAIN', confidence=50.0,
            real_score=0.5, fake_score=0.5,
            ann_fake_prob=0.5, rag_similarity=0.5,
            # sensationalism=0.0, attribution=0.0,
            topic='general', location='global',
            temporality='any',
            explanation=reason or 'Insufficient input',
            similar_articles=[],
            processing_ms=(time.time() - t0) * 1000,
        )


# ── CLI smoke-test ────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    q = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'Hyderabad rain today'
    p = TruthLensPredictor()
    p.rag.refresh()
    r = p.predict(q)
    print(f'\nQuery   : {q}')
    print(f'Verdict : {r.verdict}  ({r.confidence:.1f}%)')
    print(f'Real    : {r.real_score:.3f}  Fake: {r.fake_score:.3f}')
    print(f'ANN     : {r.ann_fake_prob:.3f}  RAG: {r.rag_similarity:.3f}')
    print(f'Temporal: {r.temporality}')
    print(f'Reason  : {r.explanation}')
