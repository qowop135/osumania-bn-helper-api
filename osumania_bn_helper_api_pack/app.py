import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests
from flask import Flask, request, jsonify

# rosu-pp-py (Rust-backed) for accurate SR/PP
# https://github.com/MaxOhn/rosu-pp-py
import rosu_pp_py as rosu

app = Flask(__name__)

TOKEN_ENV = "BN_HELPER_API_TOKEN"

# ---------------------------
# Auth (GPT Actions friendly)
# ---------------------------
def _get_auth_token() -> Optional[str]:
    tok = os.getenv(TOKEN_ENV)
    return tok.strip() if tok and tok.strip() else None

def _extract_token_from_headers(headers) -> Optional[str]:
    # GPT Actions: custom headers are limited; typically sends Authorization.
    auth = headers.get("Authorization") or headers.get("authorization")
    if auth:
        auth = auth.strip()
        m = re.match(r"^Bearer\s+(.+)$", auth, flags=re.I)
        return (m.group(1).strip() if m else auth)

    for k in ("x-api-key", "X-API-Key", "x_api_key"):
        v = headers.get(k)
        if v:
            return v.strip()
    return None

def require_auth_if_configured():
    expected = _get_auth_token()
    if not expected:
        return None
    got = _extract_token_from_headers(request.headers)
    if got != expected:
        return jsonify({"error": "Unauthorized"}), 401
    return None

# ---------------------------
# OpenAI "Sending files" helper
# ---------------------------
def _osu_text_from_openai_file_refs(openai_file_refs: Any) -> Optional[str]:
    """
    openai_file_refs: array of {name,id,mime_type,download_link} (download_link valid ~5 min)
    """
    if not isinstance(openai_file_refs, list) or not openai_file_refs:
        return None

    # pick first .osu-ish file
    picked = None
    for f in openai_file_refs:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or "").lower()
        mime = (f.get("mime_type") or "").lower()
        if name.endswith(".osu") or "osu" in name or mime in ("text/plain", "application/octet-stream"):
            picked = f
            break
    if picked is None:
        picked = openai_file_refs[0] if isinstance(openai_file_refs[0], dict) else None
    if not picked:
        return None

    url = picked.get("download_link")
    if not url:
        return None

    # Download file contents
    r = requests.get(url, timeout=15)
    r.raise_for_status()

    # .osu is UTF-8 usually; be tolerant
    return r.content.decode("utf-8", errors="replace")

def get_osu_text_from_request(data: dict) -> Optional[str]:
    osu_text = data.get("osu_text")
    if isinstance(osu_text, str) and osu_text.strip():
        return osu_text

    # OpenAI file refs parameter name must be `openaiFileIdRefs`
    refs = data.get("openaiFileIdRefs")
    try:
        return _osu_text_from_openai_file_refs(refs)
    except Exception:
        return None

# ---------------------------
# Minimal .osu parsing for BN metrics
# ---------------------------
@dataclass
class HitObj:
    t: int
    lane: int
    is_ln: bool
    end_t: int

def _parse_osu(osu_text: str) -> Tuple[int, List[HitObj]]:
    section = None
    keys = None
    objs: List[HitObj] = []

    for raw in osu_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "Difficulty" and ":" in line:
            k, v = line.split(":", 1)
            if k.strip() == "CircleSize":
                try:
                    keys = int(float(v.strip()))
                except Exception:
                    keys = None
        if section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                x = int(parts[0])
                t = int(parts[2])
                typ = int(parts[3])
            except Exception:
                continue

            if keys is None:
                keys = 4

            lane = min(keys - 1, max(0, int(x * keys / 512)))

            is_ln = (typ & 128) != 0
            end_t = t
            if is_ln and len(parts) >= 6:
                end_field = parts[5]
                end_str = end_field.split(":", 1)[0]
                try:
                    end_t = int(end_str)
                except Exception:
                    end_t = t

            objs.append(HitObj(t=t, lane=lane, is_ln=is_ln, end_t=end_t))

    if keys is None:
        keys = 4
    return keys, objs

def _window_peak_nps(times_ms: List[int], window_ms: int = 5000) -> float:
    if not times_ms:
        return 0.0
    times_ms.sort()
    q = deque()
    best = 0
    for t in times_ms:
        q.append(t)
        while q and (t - q[0]) > window_ms:
            q.popleft()
        best = max(best, len(q))
    return best / (window_ms / 1000)

def _lane_bias(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    k = len(counts)
    avg = total / k
    tv = sum(abs(c - avg) for c in counts) / (2 * total)
    return float(tv)

def _detect_jacks(objs: List[HitObj], jack_ms: int = 100):
    """
    Approx jack detector: 3+ consecutive notes in same lane where dt <= jack_ms.
    """
    by_lane: Dict[int, List[int]] = defaultdict(list)
    for o in objs:
        by_lane[o.lane].append(o.t)

    jack_count = 0
    sections = []
    for lane, ts in by_lane.items():
        ts.sort()
        run_start = None
        run_len = 1
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i-1]
            if dt <= jack_ms:
                if run_start is None:
                    run_start = ts[i-1]
                    run_len = 2
                else:
                    run_len += 1
            else:
                if run_start is not None and run_len >= 3:
                    jack_count += 1
                    sections.append({"lane": lane, "start_ms": run_start, "end_ms": ts[i-1], "notes": run_len})
                run_start = None
                run_len = 1
        if run_start is not None and run_len >= 3:
            jack_count += 1
            sections.append({"lane": lane, "start_ms": run_start, "end_ms": ts[-1], "notes": run_len})
    return int(jack_count), sections

def _chord_distribution(objs: List[HitObj]) -> Dict[str, int]:
    groups: Dict[int, int] = defaultdict(int)
    for o in objs:
        groups[o.t] += 1
    dist: Dict[str, int] = defaultdict(int)
    for size in groups.values():
        dist[f"{size}k"] += 1
    return dict(dist)

def _transition_spikes(times_ms: List[int], bucket_ms: int = 2000, spike_ratio: float = 1.75) -> int:
    if not times_ms:
        return 0
    times_ms.sort()
    t0, t1 = times_ms[0], times_ms[-1]
    if t1 <= t0:
        return 0
    buckets: Dict[int, int] = defaultdict(int)
    for t in times_ms:
        b = (t - t0) // bucket_ms
        buckets[int(b)] += 1
    spikes = 0
    max_b = max(buckets.keys())
    for b in range(1, max_b + 1):
        prev = buckets.get(b - 1, 0)
        cur = buckets.get(b, 0)
        if prev > 0 and (cur / prev) >= spike_ratio:
            spikes += 1
    return int(spikes)

def compute_metrics(osu_text: str) -> Dict[str, Any]:
    keys, objs = _parse_osu(osu_text)
    if not objs:
        return {
            "keys": keys,
            "nps_avg": 0.0,
            "nps_peak_5s": 0.0,
            "lane_bias": 0.0,
            "jack_count": 0,
            "jack_sections": [],
            "chord_dist": {},
            "ln_ratio": 0.0,
            "transition_spikes": 0,
        }

    times = sorted([o.t for o in objs])
    length_s = max(1e-6, (times[-1] - times[0]) / 1000.0)

    nps_avg = len(objs) / length_s
    nps_peak = _window_peak_nps(times, 5000)

    lane_counts = [0] * keys
    for o in objs:
        lane_counts[o.lane] += 1

    bias = _lane_bias(lane_counts)
    jack_count, jack_sections = _detect_jacks(objs, jack_ms=100)
    chord_dist = _chord_distribution(objs)
    ln_ratio = sum(1 for o in objs if o.is_ln) / max(1, len(objs))
    spikes = _transition_spikes(times)

    return {
        "keys": keys,
        "nps_avg": round(nps_avg, 4),
        "nps_peak_5s": round(nps_peak, 4),
        "lane_bias": round(bias, 4),
        "jack_count": int(jack_count),
        "jack_sections": jack_sections[:200],
        "chord_dist": chord_dist,
        "ln_ratio": round(ln_ratio, 4),
        "transition_spikes": int(spikes),
    }

# ---------------------------
# rosu-pp helpers (SR / PP)
# ---------------------------
def _parse_mods(mods: Any) -> Any:
    if mods is None:
        return ""
    if isinstance(mods, str):
        return mods
    return mods

def calc_sr_with_rosu(osu_text: str, mods: Any = "") -> Dict[str, Any]:
    # rosu supports Beatmap(content=...) + is_suspicious() checks
    m = rosu.Beatmap(content=osu_text)
    if m.is_suspicious():
        raise ValueError("Beatmap flagged as suspicious; refusing calculation.")
    keys = int(getattr(m, "cs", 0) or 0) or 4
    try:
        m.convert(rosu.GameMode.Mania, f"{keys}K")
    except Exception:
        pass

    diff = rosu.Difficulty(mods=_parse_mods(mods), lazer=False)
    attrs = diff.calculate(m)
    return {"keys": keys, "sr": float(attrs.stars)}

def calc_pp_with_rosu(osu_text: str, acc: float = 95.0, mods: Any = "") -> Dict[str, Any]:
    m = rosu.Beatmap(content=osu_text)
    if m.is_suspicious():
        raise ValueError("Beatmap flagged as suspicious; refusing calculation.")
    keys = int(getattr(m, "cs", 0) or 0) or 4
    try:
        m.convert(rosu.GameMode.Mania, f"{keys}K")
    except Exception:
        pass

    perf = rosu.Performance(
        accuracy=float(acc),
        mods=_parse_mods(mods),
        lazer=False,
        hitresult_priority=rosu.HitResultPriority.Fastest,
    )

    attrs = perf.calculate(m)
    return {"keys": keys, "pp": float(attrs.pp), "sr": float(attrs.difficulty.stars)}

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/calc_sr")
def calc_sr():
    auth_resp = require_auth_if_configured()
    if auth_resp:
        return auth_resp

    data = request.get_json(silent=True) or {}
    osu_text = get_osu_text_from_request(data)
    mods = data.get("mods", "")

    if not osu_text:
        return jsonify({"error": "Missing osu_text or openaiFileIdRefs"}), 400

    try:
        sr = calc_sr_with_rosu(osu_text, mods=mods)
        metrics = compute_metrics(osu_text)
        return jsonify({"keys": sr["keys"], "sr": sr["sr"], "details": metrics})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/calc_pp")
def calc_pp():
    auth_resp = require_auth_if_configured()
    if auth_resp:
        return auth_resp

    data = request.get_json(silent=True) or {}
    osu_text = get_osu_text_from_request(data)
    mods = data.get("mods", "")
    acc = data.get("acc", 95.0)

    if not osu_text:
        return jsonify({"error": "Missing osu_text or openaiFileIdRefs"}), 400

    try:
        out = calc_pp_with_rosu(osu_text, acc=float(acc), mods=mods)
        return jsonify(out)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/analyze_patterns")
def analyze_patterns():
    auth_resp = require_auth_if_configured()
    if auth_resp:
        return auth_resp

    data = request.get_json(silent=True) or {}
    osu_text = get_osu_text_from_request(data)
    if not osu_text:
        return jsonify({"error": "Missing osu_text or openaiFileIdRefs"}), 400

    section = data.get("section")
    if isinstance(section, str) and "-" in section:
        def to_ms(s: str) -> int:
            mm, ss = s.strip().split(":")
            return (int(mm) * 60 + int(ss)) * 1000
        try:
            a, b = section.split("-", 1)
            start_ms = to_ms(a)
            end_ms = to_ms(b)
            keys, objs = _parse_osu(osu_text)
            objs = [o for o in objs if start_ms <= o.t <= end_ms]
            if objs:
                times = sorted([o.t for o in objs])
                length_s = max(1e-6, (times[-1] - times[0]) / 1000.0)
                nps_avg = len(objs) / length_s
                nps_peak = _window_peak_nps(times, 5000)
                lane_counts = [0] * keys
                for o in objs:
                    lane_counts[o.lane] += 1
                bias = _lane_bias(lane_counts)
                jack_count, jack_sections = _detect_jacks(objs, 100)
                chord_dist = _chord_distribution(objs)
                ln_ratio = sum(1 for o in objs if o.is_ln) / max(1, len(objs))
                spikes = _transition_spikes(times)

                metrics = {
                    "keys": keys,
                    "nps_avg": round(nps_avg, 4),
                    "nps_peak_5s": round(nps_peak, 4),
                    "lane_bias": round(bias, 4),
                    "jack_count": int(jack_count),
                    "jack_sections": jack_sections[:200],
                    "chord_dist": chord_dist,
                    "ln_ratio": round(ln_ratio, 4),
                    "transition_spikes": int(spikes),
                }
            else:
                metrics = compute_metrics(osu_text)
        except Exception:
            metrics = compute_metrics(osu_text)
    else:
        metrics = compute_metrics(osu_text)

    # Risk hints (NOT hard rules)
    risks = []
    b = metrics.get("lane_bias", 0.0)
    if b >= 0.18:
        risks.append({"type": "lane_bias", "level": "HIGH", "reason": f"lane_bias={b} (unbalanced distribution)", "section": section or "all"})
    elif b >= 0.12:
        risks.append({"type": "lane_bias", "level": "MID", "reason": f"lane_bias={b}", "section": section or "all"})

    jc = metrics.get("jack_count", 0)
    if jc >= 8:
        risks.append({"type": "jack_density", "level": "HIGH", "reason": f"jack_sequences={jc}", "section": section or "all"})
    elif jc >= 3:
        risks.append({"type": "jack_density", "level": "MID", "reason": f"jack_sequences={jc}", "section": section or "all"})

    chord_dist = metrics.get("chord_dist", {}) or {}
    max_chord = 0
    for k in chord_dist.keys():
        try:
            max_chord = max(max_chord, int(k.replace("k", "")))
        except Exception:
            pass
    if max_chord >= 6:
        risks.append({"type": "large_chords", "level": "MID", "reason": f"max_chord={max_chord}", "section": section or "all"})

    return jsonify({"metrics": metrics, "risks": risks})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
