"""Scrape team lists (17 fielded players per side) for NRL matches 2018-2026."""
import re, time, sys, os, json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import threading

HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact via site form)"}
MONTHS = {m: i+1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
ALIASES = {"bulldogs": "canterbury-bankstown-bulldogs", "melbourne": "melbourne-storm",
           "cronulla": "cronulla-sutherland-sharks"}

def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))

ROW_RE = re.compile(
    r'<tr>\s*'
    r'<td><a href="/competitions/\d+">.*?</td>\s*'
    r'<td align="right">(?P<date>[^<]*)</td>\s*'
    r'<td>(?P<daytime>[^<]*)</td>\s*'
    r'<td class="team"><a href="/seasons/[^/]+/(?P<home>[^/]+)/summary\.html">[^<]*</a></td>\s*'
    r'<td class="n">(?P<hp>\d+)<!--[^>]*-->\s*</td>\s*'
    r'<td class="team"><a href="/seasons/[^/]+/(?P<away>[^/]+)/summary\.html">[^<]*</a></td>\s*'
    r'<td class="n">(?P<ap>\d+)<!--[^>]*-->\s*</td>\s*'
    r'(?P<rest>.*?)</tr>', re.S)
MATCH_ID_RE = re.compile(r'href="/matches/(\d+)"')

def get_match_index(years):
    """Return list of dicts: match_id, season, date, home, away."""
    out = []
    for year in years:
        html = fetch(f"https://www.rugbyleagueproject.org/seasons/nrl-{year}/results.html")
        cur_month = None
        for chunk in re.split(r'(?=<tr>)', html):
            m = ROW_RE.search(chunk)
            if not m:
                continue
            d = m.group("date").strip()
            dm = re.match(r'([A-Z][a-z]{2})\s+(\d+)', d)
            if dm:
                cur_month = MONTHS[dm.group(1)]; day = int(dm.group(2))
            elif d.isdigit():
                day = int(d)
            else:
                continue
            mid = MATCH_ID_RE.search(m.group("rest"))
            if not mid:
                continue
            out.append(dict(
                match_id=int(mid.group(1)), season=year,
                date=f"{year}-{cur_month:02d}-{day:02d}",
                home_team=ALIASES.get(m.group("home"), m.group("home")),
                away_team=ALIASES.get(m.group("away"), m.group("away"))))
        time.sleep(0.5)
    return out

# team list table row: home player td, home num, position th, away num, away player td
PLAYER_ROW_RE = re.compile(
    r'<tr>\s*'
    r'<td[^>]*class="name left"><a href="/players/(?P<hid>\d+)">(?P<hname>.*?)</a>[^<]*</td>\s*'
    r'<td[^>]*class="n">(?P<hnum>[^<]*)</td>\s*'
    r'<th[^>]*><abbr title="(?P<pos>[^"]+)">[^<]*</abbr></th>\s*'
    r'<td[^>]*class="n">(?P<anum>[^<]*)</td>\s*'
    r'<td[^>]*class="name"><a href="/players/(?P<aid>\d+)">(?P<aname>.*?)</a>[^<]*</td>', re.S)

def clean_name(s):
    return re.sub(r'<[^>]+>', ' ', s).strip().replace('  ', ' ')

def parse_match(match_id, html):
    sec = re.search(r'<tbody id="match_teams">(.*?)</tbody>', html, re.S)
    if not sec:
        return []
    rows = []
    for m in PLAYER_ROW_RE.finditer(sec.group(1)):
        rows.append(dict(match_id=match_id, side="home", player_id=int(m.group("hid")),
                         player=clean_name(m.group("hname")), position=m.group("pos")))
        rows.append(dict(match_id=match_id, side="away", player_id=int(m.group("aid")),
                         player=clean_name(m.group("aname")), position=m.group("pos")))
    return rows

_lock = threading.Lock()
_done = 0

def scrape_all(index, out_path, progress_path, workers=3, delay=0.35):
    global _done
    existing = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["match_id"])
                except Exception:
                    pass
    todo = [m for m in index if m["match_id"] not in existing]
    total = len(todo)
    f_out = open(out_path, "a")

    def work(meta):
        global _done
        time.sleep(delay)
        try:
            html = fetch(f"https://www.rugbyleagueproject.org/matches/{meta['match_id']}")
            players = parse_match(meta["match_id"], html)
            rec = dict(meta, players=players, n_players=len(players))
        except Exception as e:
            rec = dict(meta, players=[], n_players=0, error=str(e))
        with _lock:
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            _done += 1
            if _done % 25 == 0 or _done == total:
                with open(progress_path, "w") as pf:
                    pf.write(f"{_done}/{total}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    f_out.close()

if __name__ == "__main__":
    years = range(int(sys.argv[1]), int(sys.argv[2]) + 1)
    print("Building match index...")
    index = get_match_index(years)
    print(f"{len(index)} matches indexed")
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(index, open(os.path.join(here, "match_index.json"), "w"))
    scrape_all(index, os.path.join(here, "lineups.jsonl"), os.path.join(here, "scrape_progress.txt"))
    print("DONE")
