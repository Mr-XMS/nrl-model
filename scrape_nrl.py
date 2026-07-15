"""Scrape historical NRL results from rugbyleagueproject.org into a CSV."""
import re, time, sys
import urllib.request
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact via site form)"}
MONTHS = {m: i+1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}

def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")

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

ROUND_RE = re.compile(r'<a href="/seasons/nrl-\d+/([^/]+)/summary\.html">([^<]+)</a>\s*</th>')
VENUE_RE = re.compile(r'<a href="/venues/\d+">([^<]+)</a>')

def parse_season(html, year):
    rows = []
    # walk through table, tracking current round header and current month
    # split on <tr> chunks
    cur_round = None
    cur_month = None
    for chunk in re.split(r'(?=<tr>)', html):
        rh = re.search(r'<th colspan="11">\s*<a href="/seasons/nrl-\d+/[^"]*">([^<]+)</a>', chunk)
        if rh:
            cur_round = rh.group(1).strip()
            continue
        m = ROW_RE.search(chunk)
        if not m:
            continue
        d = m.group("date").strip()
        dm = re.match(r'([A-Z][a-z]{2})\s+(\d+)', d)
        if dm:
            cur_month = MONTHS[dm.group(1)]
            day = int(dm.group(2))
        elif d.isdigit():
            day = int(d)
        else:
            continue
        vm = VENUE_RE.search(m.group("rest"))
        rows.append({
            "season": year,
            "round": cur_round,
            "date": f"{year}-{cur_month:02d}-{day:02d}",
            "home_team": m.group("home"),
            "away_team": m.group("away"),
            "home_score": int(m.group("hp")),
            "away_score": int(m.group("ap")),
            "venue": vm.group(1) if vm else None,
        })
    return rows

def main(start, end):
    all_rows = []
    for year in range(start, end + 1):
        url = f"https://www.rugbyleagueproject.org/seasons/nrl-{year}/results.html"
        try:
            html = fetch(url)
            rows = parse_season(html, year)
            print(f"{year}: {len(rows)} matches")
            all_rows.extend(rows)
            time.sleep(1.0)  # be polite
        except Exception as e:
            print(f"{year}: FAILED ({e})")
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv("/home/claude/nrl_results.csv", index=False)
    print(f"\nTotal: {len(df)} matches -> nrl_results.csv")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
