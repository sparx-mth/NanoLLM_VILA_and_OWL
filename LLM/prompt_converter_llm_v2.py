#!/usr/bin/env python3
# prompt_converter_llm.py — minimal: sentence -> prompts (flat list)
import os, re, json, requests
from flask import Flask, request, jsonify

# ---------- config ----------
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")  # small+fast
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
MAX_PREDICT    = int(os.getenv("CONVERTER_MAX_PREDICT", "24"))
MAX_CTX        = int(os.getenv("CONVERTER_MAX_CTX", "192"))

SYSTEM_PROMPT = """
You convert a short scene description into a SINGLE list of OWL-ViT prompts.

Output STRICT JSON and NOTHING else, exactly this shape:
{"texts":[["a chair","a chair[leg]","a mug"]]}

Rules (hard):
- Lowercase, English, singular.
- Each item starts with "a " or "an ".
- NO colors or adjectives at all (drop words like white, black, yellow, long, small, big, etc.).
- NO numbers/counts and NO plurals; ignore list indices like "1.", "2.".
- Use only objects and their parts (affiliations) via brackets: head[part]. Example: "a bicycle[frame]", "a gun[barrel]".
- For “X with Y ...” patterns, keep the head X and convert Y into parts on that head, split on "and" and commas.
- Do NOT output standalone parts without a head. E.g., from “bicycle with a frame” produce "a bicycle" and "a bicycle[frame]" (NOT "a frame").
- If nothing valid is present, output {"texts":[[]]}.
"""



# ---------- tiny normalizer ----------
_COLORS = {"black","white","red","green","blue","yellow","orange","purple","pink","gray","grey","brown","silver","gold"}
_ADJ    = _COLORS | {"long","short","small","big","large","tiny","huge","tall","wide","narrow","round","square","metal","plastic","wooden"}
_ART    = {"a","an","the"}
_CONJ   = {"and", "i"}
_PRON   = {"it","them","they","he","she","this","that","these","those",
           "one","ones","its","his","her","their","my","our",
           "your", "with", "which"}
_PARTS_HOOKS = {"with","having","featuring","including"}
_REL_PREPS   = {
    "on","in","at","under","over","above","below","near","next","to","beside",
    "by","against","inside","outside","around","behind","before","after","within",
    "onto","into","upon","across","through","between","among","around", "upright",
    "mounted"
}
_VERBS = {
    "standing","sitting","lying","holding","carrying","wearing","looking",
    "using","walking","running","pointing","grabbing","placing","taking",
    "putting","standing,", "sitting,", "holding,"
}
HUMANS = {"man","woman","men","women","people","person","boy","girl"}
_STOP   = {"room","floor","ground","wall","background","of","area","space"} | _CONJ

_PART_ONLY = {
    "frame","seat","barrel","label","wheel","handle","zipper","screen","keyboard",
    "touchpad","key","button","cable","stand","seat","back","knob","eye","face"
}

_CONTAINER_HEADS = {"box","bag","bottle","pack","carton"}

_EOS_RE = re.compile(r'(?:</s>|<s>)', re.I)
_CLAUSE_SPLIT_RE = re.compile(r"[.\n;]+|,+")
_POSSESSIVE_RE = re.compile(r"\b([a-z]+)(?:'s)?\s+([a-z]+)\b", re.I)


def _from_dict_items(texts_obj_list):
    """
    Accepts [{"text":"A black","head":"speaker"}, {"text":"A white","head":"mug"}]
    → ["a speaker", "a mug"]

    Also supports variants like {"head":"bicycle","part":"frame"} → ["a bicycle","a bicycle[frame]"].
    """
    out = []
    for d in texts_obj_list or []:
        if not isinstance(d, dict):
            continue
        head = (d.get("head") or d.get("object") or d.get("noun") or "").strip()
        part = (d.get("part") or d.get("attribute") or d.get("affiliation") or "").strip()
        if head:
            out.append(f"a {head}")
            if part:
                out.append(f"a {head}[{part}]")
    return out


def _a_an(noun: str) -> str:
    noun = noun.strip()
    return ("an " if noun[:1] in "aeiou" else "a ") + noun

def _singular(w: str) -> str:
    irr = {"people":"person","men":"man","women":"woman","children":"child","mice":"mouse","bikes":"bicycle", "jeans":"jeans"}
    w = w.strip().lower()
    if w in irr: return irr[w]
    if w.endswith("ies"): return w[:-3]+"y"
    # if w.endswith("ses"): return w[:-2]
    if w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w

def _clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(?m)^\s*(\(?\d+\)?[\.\)]\s*)", "", s)      # strip "1." "(2)" prefixes
    s = re.sub(r"\s+and\s*$", "", s)                        # trailing 'and'
    s = re.sub(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_eos(s: str) -> str:
    return re.sub(r'\s+', ' ', _EOS_RE.sub('', s)).strip()

def _drop_function_words(words: list[str]) -> list[str]:
    bad = _ADJ | _ART | _PRON | _CONJ | _REL_PREPS | _VERBS
    return [w for w in words if w not in bad]

def _split_clauses(s: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(s) if c.strip()]

def _normalize_head(head: str) -> str | None:
    h = _singular(head.lower().strip())
    if h in HUMANS: h = "person"
    if h in _STOP or h in _PART_ONLY:
        return None
    return h

def _possessive_pairs(s: str) -> list[tuple[str,str]]:
    out = []
    for left, right in _POSSESSIVE_RE.findall(s):
        part = _singular(right.lower())
        if part in _PART_ONLY:
            head = _singular(left.lower())
            if head in HUMANS: head = "person"
            if head not in _STOP:
                out.append((head, part))
    return out

def _head_from_phrase(s: str) -> str | None:
    s = _clean_text(s)
    if not s: return None
    for cont in _CONTAINER_HEADS:
        if re.search(rf"\b{cont}\s+of\b", s):
            return cont
    toks = _drop_function_words(s.split())
    if not toks: return None
    head = _singular(toks[-1])
    if head in _STOP or head in _PART_ONLY:
        return None
    return head


def _parts_from_clause(clause: str) -> list[str]:
    """Extract candidate parts only; ignore relations/pronouns/articles/adjectives."""
    out = []
    clause = _clean_text(clause)
    if not clause: return out
    # split on commas or 'and'
    pieces = re.split(r"\s*(?:,| and )\s*", clause)
    for p in pieces:
        toks = _drop_function_words(p.split())
        if not toks: continue
        base = _singular(toks[-1])
        if base in _PRON or base in _ART or base in _CONJ or base in _REL_PREPS or base in _STOP:
            continue
        out.append(base)
    return out

def _drop_colors(s: str) -> str:
    # remove leading color adjectives in head or part
    # "a black speaker" -> "a speaker"
    # "a suitcase[red label]" -> "a suitcase[label]"
    try:
        head, part = s, None
        if "[" in s and s.endswith("]"):
            head, part = s.split("[", 1)
            part = part[:-1]  # drop trailing ]
        # head
        head_words = head.split()
        if len(head_words) >= 2 and head_words[1].lower() in _COLORS:
            head = head_words[0] + " " + " ".join(head_words[2:])
        # part
        if part:
            pw = part.split()
            if pw and pw[0].lower() in _COLORS:
                part = " ".join(pw[1:])
            s2 = head.strip() + "[" + part.strip() + "]"
            return s2
        return head.strip()
    except Exception:
        return s

def sanitize_texts_llm(raw_groups: list[list[str]] | list[str], *, user_prompt: str = "") -> list[list[str]]:
    """
    Normalize any LLM output + the original caption into a SINGLE group of head-only prompts:
      "a clear plastic water bottle with a green cap" -> ["a bottle"]
      "a black gun in the man's hand"                -> ["a gun"]
      "a suitcase with labels"                       -> ["a suitcase"]
      "a bicycle[frame]"                             -> ["a bicycle"]
    No colors/adjectives, no bracketed parts, no plurals.
    """

    # ---- flatten incoming groups → simple list of strings ----
    if isinstance(raw_groups, list) and raw_groups and all(isinstance(x, str) for x in raw_groups):
        raw_groups = [raw_groups]  # type: ignore
    flats: list[str] = []
    for g in (raw_groups or []):
        for p in (g or []):
            if isinstance(p, str) and p.strip():
                flats.append(p.strip())

    # also add raw caption sentences/clauses as safety net
    if user_prompt:
        up = _strip_eos(_clean_text(user_prompt))
        # if you already have _split_clauses, use it; else split on .,; and newlines
        try:
            flats.extend(_split_clauses(up))
        except NameError:
            flats.extend([seg.strip() for seg in re.split(r"[;\n\.]+", up) if seg.strip()])

    # ---- head-only extractor ----
    HOOK_OR_PREP_RE = re.compile(
        rf"\b(?:{'|'.join(sorted((_PARTS_HOOKS | _REL_PREPS | {'of'})))})\b"
    )

    def _head_only(phrase: str) -> str | None:
        s = _strip_eos(_clean_text(phrase))
        if not s:
            return None
        # Strip any bracketed parts entirely: "a bicycle[frame]" -> "a bicycle"
        if "[" in s:
            s = s.split("[", 1)[0].strip()

        # Prefer the head BEFORE any parts hook or relational preposition:
        # e.g., "a gun in the man's hand" -> consider only "a gun"
        m = HOOK_OR_PREP_RE.search(s)
        if m:
            s = s[:m.start()].strip()

        # Drop leading article and adjectives/colors
        toks = s.split()
        if toks and toks[0] in _ART:
            toks = toks[1:]
        toks = [t for t in toks if t not in _ADJ]  # colors/adjectives
        if not toks:
            return None

        # Choose the last valid token as head
        for tok in reversed(toks):
            head = _singular(tok)
            if head and head not in _STOP and head not in _PART_ONLY and head not in _VERBS\
               and head not in _REL_PREPS and head not in _ART and head != "and":
                return _a_an(head)

        return None

    # ---- build deduped output (single group) ----
    out: list[str] = []
    seen = set()
    for s in flats:
        h = _head_only(s)
        if h and h not in seen:
            seen.add(h)
            out.append(h)

    return [out]


def _flatten_strings(obj) -> list[str]:
    """
    Recursively pull out any strings from arbitrarily nested lists/tuples/dicts.
    Also understands dicts like {"head":"bicycle","part":"frame"}.
    """
    out: list[str] = []
    if obj is None:
        return out

    if isinstance(obj, str):
        out.append(obj)
        return out

    if isinstance(obj, dict):
        head = (obj.get("head") or obj.get("object") or obj.get("noun") or "")
        part = (obj.get("part") or obj.get("attribute") or obj.get("affiliation") or "")
        if head:
            head = head.strip()
            if part and isinstance(part, str) and part.strip():
                out.append(f"a {head}[{part.strip()}]")
            out.append(f"a {head}")
        # also dive into other dict values just in case
        for v in obj.values():
            out.extend(_flatten_strings(v))
        return out

    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_strings(v))
        return out

    # anything else → ignore
    return out


def _strip_enumeration(s: str) -> str:
    # remove leading "1. ", "2) ", "- " at line starts
    return re.sub(r'(?m)^\s*(?:\d+[\.\)]|-)\s*', '', s).strip()

# ---------- LLM call ----------
def llm_convert_to_texts(sentence: str):
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _strip_enumeration(_strip_eos(sentence.strip())),
        "stream": False,
        "format": "json",
        "options": {"num_predict": MAX_PREDICT, "num_ctx": MAX_CTX, "temperature": 0.1, "top_p": 0.8},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      data=json.dumps(payload),
                      headers={"Content-Type":"application/json"},
                      timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    txt = (r.json().get("response") or "").strip()

    def _try_parse(block: str) -> list[list[str]] | None:
        try:
            obj = json.loads(block)
        except Exception:
            return None
        texts = obj.get("texts") if isinstance(obj, dict) else None
        if not isinstance(texts, list):
            return None

        # 1) Already [["a …", …]]
        if texts and isinstance(texts[0], list) and all(isinstance(x, str) for x in texts[0]):
            return texts

        # 2) Flat ["a …", …]
        if all(isinstance(x, str) for x in texts):
            return [texts]

        # 3) Dict items [{"head":"…","part":"…"}, …]
        if texts and isinstance(texts[0], dict):
            flat = _flatten_strings(texts)
            return [flat] if flat else [[]]

        # 4) Arbitrary nested token soup → flatten all strings
        flat = _flatten_strings(texts)
        return [flat] if flat else [[]]

    # First pass: strict
    res = _try_parse(txt)
    if res is not None:
        return res

    # Find first JSON object
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        res = _try_parse(m.group(0))
        if res is not None:
            return res

    # Find first [[...]]
    m = re.search(r"\[\s*\[.*?\]\s*\]", txt, flags=re.S)
    if m:
        try:
            return json.loads(re.sub(r"'", '"', m.group(0)))
        except Exception:
            pass

    # Final fallback: let sanitizer mine the sentence directly
    # Return a structure compatible with sanitize_texts_llm input:
    return [[]]



# ---------- HTTP ----------
app = Flask(__name__)

@app.route("/prompts", methods=["POST"])
def prompts_only():
    """
    Request JSON: {"sentence":"..."}  (alias: {"caption":"..."})
    Response JSON: {"prompts":[ "a foo", "a foo[bar]", ... ]}
    """
    data = request.get_json(silent=True) or {}
    sentence = (data.get("sentence") or data.get("caption") or "").strip()
    sentence = _strip_enumeration(sentence)
    if not sentence:
        return jsonify({"error": "missing 'sentence'"}), 400
    try:
        raw_texts = llm_convert_to_texts(sentence)  # may return [] or [[]] on fallback
        nested = sanitize_texts_llm(raw_texts, user_prompt=sentence)
        flat = nested[0] if (isinstance(nested, list) and nested and isinstance(nested[0], list)) else (nested or [])
        return jsonify({"prompts": flat}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"upstream LLM error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

if __name__ == "__main__":
    # dev server (use gunicorn in production)
    app.run(host="0.0.0.0", port=5050, debug=False)
