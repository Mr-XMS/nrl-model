"""
Injury adjustment layer
=======================
Pulls the official NRL casualty ward feed and adjusts player ratings for
match fitness. Applied to players NAMED in the 17 (unnamed injured players
are already excluded by the team list itself - no double counting).

Adjustment rules (priors - logged so they can be calibrated over time):
  RETURNING  named, casualty list says return = this round or earlier
             -> first game back, keep 80% of rating
  IN DOUBT   named, casualty list says return is LATER than this round
             -> playing hurt or a late-withdrawal risk, keep 60% of rating
             and flag loudly

Rating multipliers only shrink toward zero (league average) - an injury
never makes a below-average player look better; adjustment is skipped
for negative ratings.
"""
import json, re, unicodedata
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
RETURNING_FACTOR = 0.80
IN_DOUBT_FACTOR = 0.60

NICK_TO_SLUG = {
    "Broncos": "brisbane-broncos", "Raiders": "canberra-raiders",
    "Bulldogs": "canterbury-bankstown-bulldogs", "Sharks": "cronulla-sutherland-sharks",
    "Dolphins": "dolphins", "Titans": "gold-coast-titans",
    "Sea Eagles": "manly-warringah-sea-eagles", "Storm": "melbourne-storm",
    "Knights": "newcastle-knights", "Cowboys": "north-queensland-cowboys",
    "Eels": "parramatta-eels", "Panthers": "penrith-panthers",
    "Rabbitohs": "south-sydney-rabbitohs", "Dragons": "st-george-illawarra-dragons",
    "Roosters": "sydney-roosters", "Warriors": "warriors", "Wests Tigers": "wests-tigers",
}

def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())

def get_casualty_list():
    """Return {(team_slug, normalised_name): {injury, expected_return, return_round}}"""
    req = urllib.request.Request("https://www.nrl.com/casualty-ward/data", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = {}
    for c in data.get("casualties", []):
        team = NICK_TO_SLUG.get(c.get("teamNickname"))
        if not team:
            continue
        exp = (c.get("expectedReturn") or "").strip()
        m = re.search(r"Round\s+(\d+)", exp, re.I)
        out[(team, _norm(c["firstName"] + c["lastName"]))] = dict(
            name=f"{c['firstName']} {c['lastName']}",
            injury=c.get("injury", "?"),
            expected_return=exp or "unknown",
            return_round=int(m.group(1)) if m else None)
    return out

def injury_status(team_slug, first, last, current_round, casualties):
    """Return (status, factor, info) for a named player.
    status: None | 'returning' | 'in_doubt'"""
    rec = casualties.get((team_slug, _norm(first + last)))
    if rec is None:
        return None, 1.0, None
    rr = rec["return_round"]
    if rr is None:
        # 'Season', 'TBC', 'Indefinite' etc. but somehow named -> treat as in doubt
        return "in_doubt", IN_DOUBT_FACTOR, rec
    if rr > current_round:
        return "in_doubt", IN_DOUBT_FACTOR, rec
    return "returning", RETURNING_FACTOR, rec

if __name__ == "__main__":
    cw = get_casualty_list()
    print(f"{len(cw)} players on the official casualty ward:\n")
    by_team = {}
    for (team, _), rec in cw.items():
        by_team.setdefault(team, []).append(rec)
    for team in sorted(by_team):
        print(team)
        for rec in by_team[team]:
            print(f"   {rec['name']:26s} {rec['injury']:22s} -> {rec['expected_return']}")
