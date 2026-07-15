"""
Weather features for the NRL model
==================================
- add_wet_column(results): wet flag (>=5mm rain at venue city) for every
  historical match, using weather.csv (auto-topped-up from the Open-Meteo
  archive when results run past the stored weather).
- forecast_rain(city, date): expected precipitation (mm) for an upcoming
  fixture from the Open-Meteo forecast API (16-day horizon).
"""
import json, os, re, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER = os.path.join(HERE, "weather.csv")
WET_MM = 5.0

CITIES = dict(sydney=(-33.87,151.21), brisbane=(-27.47,153.03), melbourne=(-37.81,144.96),
    auckland=(-36.85,174.76), townsville=(-19.26,146.82), newcastle=(-32.93,151.78),
    wollongong=(-34.42,150.89), canberra=(-35.28,149.13), goldcoast=(-28.02,153.40),
    perth=(-31.95,115.86), darwin=(-12.46,130.84), cairns=(-16.92,145.77),
    centralcoast=(-33.43,151.34), mackay=(-21.14,149.19), vegas=(36.17,-115.14),
    christchurch=(-43.53,172.64), wellington=(-41.29,174.78), hamilton=(-37.79,175.28),
    sunshine=(-26.65,153.07), redcliffe=(-27.23,153.11), bathurst=(-33.42,149.58),
    tamworth=(-31.09,150.93), dubbo=(-32.24,148.60), wagga=(-35.11,147.37),
    mudgee=(-32.59,149.59), rockhampton=(-23.38,150.51), coffs=(-30.30,153.11),
    portmoresby=(-9.44,147.18), toowoomba=(-27.56,151.95), gladstone=(-23.84,151.26),
    bundaberg=(-24.87,152.35), albury=(-36.08,146.92))

VENUE_CITY = [('suncorp','brisbane'),('lang park','brisbane'),('anz','sydney'),('accor','sydney'),
 ('stadium australia','sydney'),('aami','melbourne'),('olympic park','melbourne'),
 ('allianz','sydney'),('sfs','sydney'),('scg','sydney'),('cbus','goldcoast'),
 ('skilled','goldcoast'),('centrebet','goldcoast'),('robina','goldcoast'),
 ('gio','canberra'),('canberra','canberra'),('mt smart','auckland'),('go media','auckland'),
 ('mcd. jones','newcastle'),('mcdonald jones','newcastle'),('hunter','newcastle'),
 ('win jubilee','sydney'),('jubilee','sydney'),('kogarah','sydney'),('win','wollongong'),
 ('1300smiles','townsville'),('qld c. bank','townsville'),('dairy farmers','townsville'),
 ('toyota','townsville'),('brookvale','sydney'),('4 pines','sydney'),('lottoland','sydney'),
 ('commbank','sydney'),('bankwest','sydney'),('parramatta','sydney'),('pirtek','sydney'),
 ('cua','sydney'),('pepper','sydney'),('panthers','sydney'),('bluebet','sydney'),
 ('campbelltown','sydney'),('leichhardt','sydney'),('netstrata','sydney'),
 ('remondis','sydney'),('pointsbet','sydney'),('southern cross','sydney'),
 ('kayo','redcliffe'),('moreton','redcliffe'),('cent','centralcoast'),('gosford','centralcoast'),
 ('industree','centralcoast'),('optus','perth'),('hbf','perth'),('tio','darwin'),
 ('barlow','cairns'),('cazalys','cairns'),('bb print','mackay'),('mackay','mackay'),
 ('allegiant','vegas'),('browne','portmoresby'),('santos','portmoresby'),
 ('scully','tamworth'),('glen willow','mudgee'),('carrington','bathurst'),
 ('mcdonalds','wagga'),('salter','bundaberg'),('marley brown','gladstone'),
 ('clive berghofer','toowoomba'),('c.ex','coffs')]

TEAM_CITY = {'brisbane-broncos':'brisbane','canberra-raiders':'canberra',
 'canterbury-bankstown-bulldogs':'sydney','cronulla-sutherland-sharks':'sydney',
 'dolphins':'redcliffe','gold-coast-titans':'goldcoast','manly-warringah-sea-eagles':'sydney',
 'melbourne-storm':'melbourne','newcastle-knights':'newcastle',
 'north-queensland-cowboys':'townsville','parramatta-eels':'sydney','penrith-panthers':'sydney',
 'south-sydney-rabbitohs':'sydney','st-george-illawarra-dragons':'sydney',
 'sydney-roosters':'sydney','warriors':'auckland','wests-tigers':'sydney'}


def venue_to_city(venue, home_team):
    if isinstance(venue, str):
        vl = venue.lower()
        for k, c in VENUE_CITY:
            if k in vl:
                return c
    return TEAM_CITY.get(home_team, "sydney")


def nrl_city_to_key(venue_city_str, home_team):
    """Map nrl.com's venueCity string (e.g. 'Gold Coast') to our city key."""
    if venue_city_str:
        key = re.sub(r"[^a-z]", "", venue_city_str.lower())
        aliases = {"lasvegas": "vegas", "sunshinecoast": "sunshine",
                   "goldcoast": "goldcoast", "centralcoast": "centralcoast"}
        key = aliases.get(key, key)
        if key in CITIES:
            return key
    return TEAM_CITY.get(home_team, "sydney")


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _topup_weather(W, need_until):
    """Fetch missing recent days from the Open-Meteo archive (best effort)."""
    start = (W.date.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = need_until.strftime("%Y-%m-%d")
    frames = [W]
    for city, (lat, lon) in CITIES.items():
        try:
            d = _get_json(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
                          f"&longitude={lon}&start_date={start}&end_date={end}"
                          f"&daily=precipitation_sum&timezone=auto")
            frames.append(pd.DataFrame(dict(city=city,
                date=pd.to_datetime(d["daily"]["time"]),
                rain_mm=d["daily"]["precipitation_sum"])))
        except Exception:
            pass
    W = pd.concat(frames).drop_duplicates(["city", "date"])
    W.to_csv(WEATHER, index=False)
    return W


def add_wet_column(results):
    """Return a 0/1 wet array aligned to the results dataframe."""
    if not os.path.exists(WEATHER):
        return np.zeros(len(results), dtype=int)
    W = pd.read_csv(WEATHER, parse_dates=["date"])
    if results.date.max() > W.date.max():
        try:
            W = _topup_weather(W, results.date.max())
        except Exception:
            pass
    W = W.set_index(["city", "date"]).rain_mm
    wet = []
    for r in results.itertuples():
        city = venue_to_city(r.venue, r.home_team)
        mm = W.get((city, r.date), 0.0)
        wet.append(1 if (mm is not None and mm >= WET_MM) else 0)
    return np.array(wet, dtype=int)


def forecast_rain(city_key, date):
    """Forecast precipitation (mm) for a city on a date, or None if unavailable."""
    try:
        lat, lon = CITIES[city_key]
        d = date.strftime("%Y-%m-%d")
        j = _get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                      f"&daily=precipitation_sum&timezone=auto&start_date={d}&end_date={d}")
        return float(j["daily"]["precipitation_sum"][0])
    except Exception:
        return None
