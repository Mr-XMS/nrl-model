"""Backfill per-player match statistics from nrl.com match centres."""
import json, os, sys, time, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_stats.jsonl")

def get_json(url, retries=3):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))

def collect_match_urls(season):
    urls = []
    for rnd in range(1, 32):
        try:
            d = get_json(f"https://www.nrl.com/draw/data?competition=111&season={season}&round={rnd}")
        except Exception:
            continue
        fx = [f for f in d.get("fixtures", []) if f.get("type") == "Match"]
        if not fx:
            if rnd > 27:
                break
            continue
        for f in fx:
            if f.get("matchState") in ("FullTime", "PostGame"):
                urls.append((season, f.get("roundTitle"), f["matchCentreUrl"]))
        time.sleep(0.2)
    return urls

_lock = threading.Lock()
_done = [0]

def scrape_stats(entries, out_path=OUT, workers=4, delay=0.25):
    seen = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["url"])
                except Exception:
                    pass
    todo = [e for e in entries if e[2] not in seen]
    total = len(todo)
    fo = open(out_path, "a")

    def work(entry):
        season, rnd, url = entry
        time.sleep(delay)
        try:
            mc = get_json("https://www.nrl.com" + url.rstrip("/") + "/data")
            rec = dict(url=url, season=season, round=rnd,
                       kickoff=mc.get("startTime"),
                       home=mc["homeTeam"]["nickName"], away=mc["awayTeam"]["nickName"],
                       home_score=mc["homeTeam"].get("score"),
                       away_score=mc["awayTeam"].get("score"), players=[])
            names = {}
            for side in ("homeTeam", "awayTeam"):
                for p in (mc.get(side, {}).get("players") or []):
                    names[p["playerId"]] = dict(
                        name=f"{p['firstName']} {p['lastName']}",
                        position=p.get("position"), number=p.get("number"))
            st = (mc.get("stats") or {}).get("players") or {}
            for side_key, side in (("homeTeam", "home"), ("awayTeam", "away")):
                for row in st.get(side_key, []):
                    info = names.get(row.get("playerId"), {})
                    rec["players"].append(dict(row, side=side, **info))
        except Exception as e:
            rec = dict(url=url, season=season, round=rnd, error=str(e), players=[])
        with _lock:
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            _done[0] += 1
            if _done[0] % 50 == 0:
                print(f"{_done[0]}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fo.close()
    print(f"done: {_done[0]} scraped")

if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]]
    entries = []
    for s in seasons:
        e = collect_match_urls(s)
        print(f"{s}: {len(e)} completed matches")
        entries.extend(e)
    scrape_stats(entries)
