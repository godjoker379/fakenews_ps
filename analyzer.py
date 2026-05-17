"""
analyzer.py — Linguistic analysis for TruthLens

Provides:
  clean_text()          — normalise text for embedding
  detect_topic()        — classify into topic buckets
  detect_location()     — geo-scope detection
  detect_temporal()     — today / yesterday / tomorrow / week / month / recent / any
  check_negation()      — is the query a debunk / fact-check request?
  get_predicates()      — extract event verbs
  get_linguistic_score() — combined dict used by predictor
"""

import re
from datetime import datetime, timedelta

# ── Keyword Banks ─────────────────────────────────────────────
SOURCE_WORDS = [
    'according to', 'reported by', 'said', 'told', 'announced', 'confirmed',
    'stated', 'officials said', 'spokesperson', 'in a statement',
    'associated press', 'reuters', 'sources say', 'government said',
    'published by', 'as per', 'citing', 'quoting',
]

SENSATIONAL_WORDS = [
    'shocking', 'bombshell', 'explosive', 'incredible', 'mind-blowing',
    'what really happened', 'conspiracy', 'hoax', 'cover-up', 'secret',
    "they don't want you to know", 'wake up', 'exposed', 'breaking',
    'you wont believe', 'truth revealed', 'hidden agenda',
]

FAKE_HARD_PHRASES = [
    'mind control', 'microchip vaccine', 'deep state', 'flat earth',
    'crisis actor', 'plandemic', 'chemtrail', 'illuminati', 'new world order',
    'reptilian', 'george soros controls', 'bill gates microchip',
    'scamdemic', 'globalist agenda', '5g causes', 'vaccines cause autism',
]

MAJOR_EVENT_VERBS = [
    'died', 'death', 'killed', 'arrested', 'won', 'lost', 'resigned',
    'hospitalized', 'guilty', 'innocent', 'banned', 'launched', 'crashed',
    'elected', 'fired', 'promoted', 'merged', 'acquired', 'collapsed',
    'attacked', 'sentenced', 'charged', 'indicted',
]

# ── Topic keywords ────────────────────────────────────────────
TOPIC_KEYWORDS: dict[str, list[str]] = {
    'weather': [
        'weather', 'rain', 'storm', 'forecast', 'imd', 'celsius',
        'flood', 'cyclone', 'monsoon', 'temperature', 'humidity',
        'drought', 'heatwave', 'lightning', 'thunderstorm', 'cloudy',
        'rainfall', 'precipitation', 'wind', 'alert', 'warning',
        'sunny', 'overcast', 'mist', 'fog', 'cold wave', 'heat wave',
        'yellow alert', 'orange alert', 'red alert', 'imd forecast',
    ],
    'sports': [
        'cricket', 'football', 'ipl', 'score', 'team', 'match',
        'player', 'tournament', 'fifa', 'olympic', 'champion',
        'test match', 'odi', 't20', 'batting', 'bowling', 'goal',
        'league', 'cup', 'trophy', 'series', 'innings', 'wicket',
        'kabaddi', 'hockey', 'badminton', 'tennis',
    ],
    'telangana': [
        'telangana', 'hyderabad', 'ghmc', 'kcr', 'ktr', 'revanth',
        'warangal', 'nizamabad', 'karimnagar', 'khammam', 'brs',
        'trs', 'secunderabad', 'charminar', 'hussain sagar',
        'cyberabad', 'kukatpally', 'lb nagar', 'hanuman junction',
        'jubilee hills', 'banjara hills', 'gachibowli', 'hitec city',
        'ameerpet', 'dilsukhnagar', 'uppal', 'sainikpuri',
        'v6', 'tv9 telugu', 'sakshi', 'abn', 'ntv telugu',
    ],
    'politics': [
        'election', 'government', 'minister', 'vote', 'policy',
        'bjp', 'congress', 'parliament', 'modi', 'rahul',
        'assembly', 'loksabha', 'rajyasabha', 'governor', 'cm',
        'opposition', 'party', 'candidate', 'manifesto', 'rally',
        'corruption', 'scam', 'protest', 'bypolls', 'mla', 'mp',
    ],
    'health': [
        'health', 'vaccine', 'covid', 'hospital', 'doctor', 'virus',
        'disease', 'medicine', 'treatment', 'surgery', 'patient',
        'pandemic', 'epidemic', 'who', 'icmr', 'aiims',
        'prescription', 'drug', 'clinical', 'symptoms', 'mpox',
    ],
    'crime': [
        'murder', 'robbery', 'theft', 'rape', 'assault', 'police',
        'arrested', 'fir', 'case', 'court', 'judge', 'verdict',
        'accused', 'criminal', 'gang', 'kidnap', 'drug bust',
        'encounter', 'bail', 'chargesheet',
    ],
    'business': [
        'stock', 'market', 'economy', 'gdp', 'inflation', 'rbi',
        'bank', 'rupee', 'dollar', 'sensex', 'nifty', 'company',
        'startup', 'merger', 'acquisition', 'profit', 'loss',
        'budget', 'tax', 'gst', 'revenue', 'investment', 'sebi',
    ],
    'technology': [
        'ai', 'artificial intelligence', 'robot', 'software',
        'hardware', 'app', 'mobile', 'internet', 'cyber', 'hack',
        'chatgpt', 'smartphone', 'tesla', 'elon', 'space',
        'satellite', 'isro', 'nasa', 'launch', 'orbit', '5g',
    ],
    'world': [
        'usa', 'america', 'china', 'russia', 'ukraine', 'war',
        'un', 'nato', 'europe', 'iran', 'israel', 'pakistan',
        'afghanistan', 'geopolitics', 'sanctions', 'diplomacy',
        'g20', 'g7', 'imf', 'world bank',
    ],
}

# ── Geography ─────────────────────────────────────────────────
GEOGRAPHY_KEYWORDS: dict[str, list[str]] = {
    'hyderabad': [
        'hyderabad', 'hyd', 'secunderabad', 'cyberabad',
        'ghmc', 'charminar', 'hussain sagar', 'jubilee hills',
        'banjara hills', 'hitec city', 'gachibowli', 'kukatpally',
        'lb nagar', 'dilsukhnagar', 'uppal', 'ameerpet',
        'madhapur', 'kondapur', 'miyapur', 'kompally',
    ],
    'telangana': [
        'telangana', 'ts', 'warangal', 'nizamabad', 'karimnagar',
        'khammam', 'nalgonda', 'mahabubnagar', 'adilabad',
        'sangareddy', 'medak', 'siddipet', 'suryapet',
        'mancherial', 'jagtial', 'rajanna', 'nagarkurnool',
    ],
    'mumbai': [
        'mumbai', 'bombay', 'maharashtra', 'pune', 'thane',
        'navi mumbai', 'bmc', 'nashik', 'aurangabad',
    ],
    'delhi': [
        'delhi', 'ncr', 'new delhi', 'noida', 'gurgaon',
        'faridabad', 'ghaziabad', 'mcd', 'ndmc',
    ],
    'bangalore': ['bangalore', 'bengaluru', 'karnataka', 'mysuru', 'bbmp'],
    'chennai':   ['chennai', 'tamil nadu', 'madras', 'coimbatore', 'madurai'],
    'kolkata':   ['kolkata', 'west bengal', 'calcutta', 'howrah'],
    'india':     ['india', 'indian', 'bharat', 'modi', 'new delhi'],
}

# ── Temporal markers (expanded) ───────────────────────────────
_TODAY_WORDS    = [
    'today', 'now', 'latest', 'just', 'breaking', 'current', 'live',
    'right now', 'this morning', 'this evening', 'this afternoon',
    'tonight', 'aaj', 'abhi', 'इस वक्त',
]
_TOMORROW_WORDS = [
    'tomorrow', 'kal', 'next day', 'day after', 'coming day',
    'tomorrow morning', 'tomorrow evening', 'tomorrow night',
]
_YEST_WORDS     = [
    'yesterday', 'last night', 'last evening', 'kal raat',
    'previous day', 'day before',
]
_RECENT_WORDS   = [
    'this week', 'recently', 'last week', 'past few days',
    'few hours ago', 'past 24 hours', 'past 48 hours',
    'is week', 'this month', 'last month',
]
_WEEK_WORDS     = [
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'this week', 'next week', 'last week',
    'weekday', 'weekend', 'weekly',
]
_MONTH_WORDS    = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'this month', 'next month', 'last month', 'monthly',
]

# ── Debunk phrases ────────────────────────────────────────────
_DEBUNK_PHRASES = [
    'is this fake', 'is this true', 'is this real',
    'fact check', 'is it true that', 'rumour that',
    'hoax about', 'debunked', 'this is false',
    'misinformation about', 'is it fake',
]


# ── Public API ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise text for embedding: remove URLs, HTML, excess punctuation."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def detect_topic(text: str) -> str:
    """Return highest-scoring topic label, or 'general'."""
    txt = text.lower()
    scores = {
        topic: sum(1 for kw in kws if kw in txt)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'general'


def detect_location(text: str) -> str:
    """Return primary geographic scope, or 'global'."""
    txt = text.lower()
    scores = {
        loc: sum(1 for kw in kws if kw in txt)
        for loc, kws in GEOGRAPHY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'global'


def detect_temporal(text: str) -> str:
    """
    Classify query time-scope as:
      'today' | 'tomorrow' | 'yesterday' | 'week' | 'month' | 'recent' | 'any'
    Order matters — more specific checks first.
    """
    txt = text.lower()
    if any(k in txt for k in _TODAY_WORDS):     return 'today'
    if any(k in txt for k in _TOMORROW_WORDS):  return 'tomorrow'
    if any(k in txt for k in _YEST_WORDS):      return 'yesterday'
    if any(k in txt for k in _MONTH_WORDS):     return 'month'
    if any(k in txt for k in _WEEK_WORDS):      return 'week'
    if any(k in txt for k in _RECENT_WORDS):    return 'recent'
    return 'any'


def temporal_label(temporality: str) -> str:
    """Human-friendly label for a temporal category."""
    now = datetime.now()
    labels = {
        'today':     f"today ({now.strftime('%A, %d %b %Y')})",
        'tomorrow':  f"tomorrow ({(now + timedelta(days=1)).strftime('%A, %d %b %Y')})",
        'yesterday': f"yesterday ({(now - timedelta(days=1)).strftime('%A, %d %b %Y')})",
        'week':      f"this week (week of {now.strftime('%d %b %Y')})",
        'month':     f"this month ({now.strftime('%B %Y')})",
        'recent':    'recent (last 48h)',
        'any':       'any time',
    }
    return labels.get(temporality, temporality)


def check_negation(text: str) -> bool:
    """Return True if the query is itself a debunking / fact-check request."""
    tl = text.lower()
    return any(p in tl for p in _DEBUNK_PHRASES)


def get_predicates(text: str) -> list[str]:
    """Extract event-verb predicates from the query."""
    tl = text.lower()
    return [v for v in MAJOR_EVENT_VERBS if f' {v}' in f' {tl}']


def get_linguistic_score(text: str) -> dict:
    """
    Returns:
        sensationalism : float 0-1
        attribution    : float 0-1
        topic          : str
        hard_hit       : bool
        temporality    : str
    """
    tl        = text.lower()
    sens_hits = sum(1 for p in SENSATIONAL_WORDS if p in tl)
    hard_hit  = any(p in tl for p in FAKE_HARD_PHRASES)
    attr_hits = sum(1 for w in SOURCE_WORDS if w in tl)

    return {
        # 'sensationalism': min((sens_hits * 0.25) + (0.4 if hard_hit else 0.0), 1.0),
        # 'attribution':    min(attr_hits / max(len(SOURCE_WORDS), 1) * 4.0, 1.0),
        'topic':          detect_topic(text),
        'hard_hit':       hard_hit,
        'temporality':    detect_temporal(text),
    }
