"""
app.py — TruthLens Flask API

Routes
──────
GET  /              → HTML frontend (index.html)
GET  /api/status    → system health + RAG stats  (always instant)
POST /api/predict   → { "query": "..." } → prediction JSON
POST /api/refresh   → force RAG index rebuild

Startup Strategy
────────────────
Flask starts IMMEDIATELY so the browser connects in < 1 s.
Model loading + RSS seeding run concurrently in a background thread.
_init_msg is updated at every stage so the UI shows live progress.
/api/predict blocks (up to 60 s) if called before init finishes.
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from predictor import TruthLensPredictor

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder = os.path.join(BASE_DIR, 'templates'),
    static_folder   = None,
)
CORS(app)

# ── Global state ──────────────────────────────────────────────
_predictor:  TruthLensPredictor | None = None
_init_msg:   str   = 'Starting up…'
_init_stage: int   = 0   # 0=idle 1=encoder 2=ann 3=rag 4=ready 5=error
_init_ready  = threading.Event()
_init_lock   = threading.Lock()
_start_time  = time.time()

AUTO_REFRESH_S = int(os.getenv('AUTO_REFRESH_S', '600'))

_STAGE_LABELS = {
    0: 'Starting up…',
    1: 'Loading sentence encoder (all-MiniLM-L6-v2)…',
    2: 'ANN model ready — seeding RAG index…',
    3: 'Fetching live RSS feeds…',
    4: 'Ready',
}


def _set_stage(stage: int, msg: str | None = None) -> None:
    global _init_msg, _init_stage
    _init_stage = stage
    _init_msg   = msg or _STAGE_LABELS.get(stage, '')
    log.info('[INIT stage %d] %s', stage, _init_msg)


# ── Background init ───────────────────────────────────────────
def _init_predictor_bg() -> None:
    """
    Load encoder + ANN, then seed RAG.
    Flask is already serving while this runs — UI polls /api/status.
    """
    global _predictor
    try:
        _set_stage(1)
        pred = TruthLensPredictor(
            model_path  = os.getenv('MODEL_PATH', 'ann.pth'),
            embed_model = 'all-MiniLM-L6-v2',
        )

        _set_stage(3)
        n = pred.rag.refresh()
        log.info('RAG seeded: %d articles.', n)

        with _init_lock:
            _predictor = pred
        _set_stage(4)
        _init_ready.set()
        log.info('TruthLens ready in %.1f s.', time.time() - _start_time)

    except Exception as exc:
        _set_stage(5, f'Init error: {exc}')
        _init_ready.set()   # unblock any waiting /api/predict
        log.exception('Predictor init failed')


def _auto_refresh_loop() -> None:
    """Daemon: re-fetch RSS every AUTO_REFRESH_S seconds after init."""
    _init_ready.wait()
    while True:
        time.sleep(AUTO_REFRESH_S)
        try:
            with _init_lock:
                pred = _predictor
            if pred:
                n = pred.rag.refresh()
                log.info('Auto-refresh: %d articles.', n)
        except Exception as exc:
            log.warning('Auto-refresh error: %s', exc)


# ── Error handler ─────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    log.exception('Unhandled exception')
    return jsonify(error=str(e), type=type(e).__name__), 500


# ── GET /api/status ───────────────────────────────────────────
@app.route('/api/status')
def api_status():
    with _init_lock:
        ready = _predictor is not None
        pred  = _predictor

    payload = {
        'ready':      ready,
        'stage':      _init_stage,
        'status_msg': _init_msg,
        'uptime_s':   int(time.time() - _start_time),
        'timestamp':  datetime.now(timezone.utc).isoformat(),
    }
    if ready and pred:
        try:
            payload['index_size']   = pred.rag.index_size
            payload['last_refresh'] = pred.rag.last_refresh
        except Exception as e:
            payload['rag_err'] = str(e)

    return jsonify(payload)


# ── POST /api/predict ─────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Wait up to 60 s for init — releases immediately once ready
    if not _init_ready.wait(timeout=60):
        return jsonify({'error': 'Engine still initialising — please try again shortly.'}), 503

    with _init_lock:
        pred = _predictor
    if pred is None:
        return jsonify({'error': 'Engine initialisation failed. Check server logs.'}), 503

    body  = request.get_json(silent=True) or {}
    query = (body.get('query') or '').strip()

    if not query:
        return jsonify({'error': 'Field "query" is required and must not be empty.'}), 400
    if len(query) > 2000:
        return jsonify({'error': 'Query exceeds 2000-character limit.'}), 400

    try:
        return jsonify(pred.predict(query).to_dict())
    except Exception as exc:
        log.exception('Prediction error')
        return jsonify({'error': str(exc)}), 500


# ── POST /api/refresh ─────────────────────────────────────────
@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    with _init_lock:
        pred = _predictor
    if pred is None:
        return jsonify({'error': 'Not ready yet.'}), 503
    t0 = time.time()
    n  = pred.rag.refresh()
    return jsonify({
        'status':     'ok',
        'articles':   n,
        'duration_s': round(time.time() - t0, 2),
    })


# ── GET / → frontend ─────────────────────────────────────────
@app.route('/')
def serve_frontend():
    try:
        return render_template('index.html')
    except Exception as exc:
        log.error('render_template failed: %s', exc)
        return (
            '<h1>TruthLens</h1>'
            f'<p>Frontend error: {exc}</p>'
            '<p>Make sure <code>templates/index.html</code> exists.</p>'
        ), 500


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    """
    Flask starts first — browser connects in < 1 s.
    Model loading + RAG seeding run concurrently in background threads.
    """
    threading.Thread(target=_init_predictor_bg, daemon=True, name='init').start()
    threading.Thread(target=_auto_refresh_loop, daemon=True, name='refresh').start()

    port = int(os.getenv('PORT', '5010'))
    log.info('Flask on http://0.0.0.0:%d  (engine loading in background…)', port)
    app.run(
        host        = '0.0.0.0',
        port        = port,
        debug       = False,
        threaded    = True,
        use_reloader= False,
    )


if __name__ == '__main__':
    main()