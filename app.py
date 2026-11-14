import inspect
import json
import math
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as html_escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3.util.connection as urllib3_conn

# Bazı ağlarda IPv6 (AAAA) DNS sorgusu ~10 sn takılıyor; HTTP isteklerinde yalnızca IPv4 kullan
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

# set_page_config ilk Streamlit komutu olmalı
st.set_page_config(
    page_title="SPACE GRAVE | SP4C3 GR4V3 - NASA Space Apps JPL Command Center",
    page_icon="🛰️",
    layout="wide"
)

try:
    from sgp4.api import Satrec, SatrecArray, jday
    from scipy.spatial import cKDTree
except ImportError:
    st.error("Eksik paketler var. Terminalde çalıştır: pip install streamlit sgp4 scipy numpy pandas plotly requests")
    st.stop()

RE = 6378.137  # km, Earth equatorial radius
WGS84_F = 1 / 298.257223563  # Dünya basıklığı
CREWED_PREFIXES = ("ISS", "CSS", "TIANGONG")


# Streamlit sürümleri arası uyumluluk:
# yeni sürümlerde use_container_width kaldırıldı -> width="stretch"
def stretch(fn):
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


# ANTI-BAN HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

# FALLBACK TLE VERİSİ
FALLBACK_TLE = """ISS (ZARYA)
1 25544U 98067A   23304.53761765  .00016717  00000-0  30043-3 0  9997
2 25544  51.6416 270.8359 0001402 121.3653 286.0718 15.50150914422687
HST (HUBBLE)
1 20580U 90037B   23304.53580556  .00001099  00000-0  48316-4 0  9991
2 20580  28.4695 244.4019 0002271 270.6277 159.2064 15.08779951465225
TIANGONG
1 48274U 21035A   23304.51528935  .00011500  00000-0  17529-3 0  9997
2 48274  41.4746 308.8687 0002213 133.0232 236.4385 15.59751992143493
"""

def _lang_from_url():
    # Navigasyondaki bağlantı sayfayı yenilediği için seçili dil ?lang= ile taşınır
    try:
        return "en" if str(st.query_params.get("lang", "")).lower() == "en" else "tr"
    except Exception:
        return "tr"


# --- SESSION STATE ---
for key, default in {
    "intro_passed": False,
    "lang": _lang_from_url(),
    "active_page": "📊 General Status Overview",
    "defense_active": False,
    "conj_df": None,
    "conj_meta": None,
    "reentry": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "bot_open" not in st.session_state:
    st.session_state.bot_open = False
if "bot_log" not in st.session_state:
    # Kayıtlar dilden bağımsız tutulur (soru kimliği); gösterim anında seçili dile göre üretilir
    st.session_state.bot_log = [{"r": "bot", "g": True}]


# --- DİL (TÜR / ENG) ---
def T(tr, en):
    return tr if st.session_state.lang == "tr" else en


def _keyed(key):
    # st.container(key=...) eski Streamlit sürümlerinde yok -> normal kutu olarak çalışır
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def _bot_ph():
    return T("💬 Başka bir soru seç…", "💬 Pick another question…")


def set_lang(lang):
    st.session_state.lang = lang
    st.session_state["sg_bot_pick"] = _bot_ph()


def _i18n_html(html, pairs):
    # İngilizce yazılmış gömülü (iframe) sayfaları Türkçe moduna çevirir
    if st.session_state.lang != "tr":
        return html
    for a, b in pairs:
        html = html.replace(a, b)
    return html


def render_lang_switch(right, bottom):
    """Sağ altta sabit ENG / TÜR kutucukları (giriş ve canlı izleme sayfasında ortak)."""
    active = "sg_lang_en" if st.session_state.lang == "en" else "sg_lang_tr"
    with _keyed("sg_lang"):
        st.button("ENG", key="sg_lang_en", on_click=set_lang, args=("en",))
        st.button("TÜR", key="sg_lang_tr", on_click=set_lang, args=("tr",))
    st.markdown(f"""
    <style>
    .st-key-sg_lang {{ position:fixed; right:{right}; bottom:{bottom}; z-index:1000000; display:flex;
      flex-direction:row !important; width:auto !important; gap:6px !important; padding:5px; border-radius:12px;
      background:rgba(6,12,30,.84); border:1px solid rgba(56,189,248,.38); backdrop-filter:blur(12px);
      -webkit-backdrop-filter:blur(12px); box-shadow:0 8px 28px rgba(0,0,0,.55); }}
    .st-key-sg_lang > div {{ width:auto !important; }}
    .st-key-sg_lang .stButton > button {{ min-height:30px; height:30px; padding:0 14px; border-radius:8px;
      background:transparent; border:1px solid transparent; box-shadow:none; }}
    .st-key-sg_lang .stButton > button p {{ font-size:12px; font-weight:700; letter-spacing:.12em; color:#94a3b8; margin:0; }}
    .st-key-sg_lang .stButton > button:hover {{ border-color:#38bdf8; }}
    .st-key-{active} .stButton > button {{ background:rgba(56,189,248,.22); border-color:#38bdf8; }}
    .st-key-{active} .stButton > button p {{ color:#fff; }}
    </style>
    """, unsafe_allow_html=True)


# --- STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0b1329 0%, #040817 50%, #000000 100%);
        color: #f3f4f6;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] { display: none; }
    .nasa-glass-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(7, 11, 25, 0.99) 100%);
        backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(0, 51, 102, 0.8);
        padding: 22px; border-radius: 20px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(11, 75, 148, 0.15);
        text-align: center; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; margin-bottom: 10px;
    }
    .fin-card {
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(16, 185, 129, 0.1) !important;
    }
    .metric-title { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }
    .metric-value { color: #38bdf8; font-size: 26px; font-weight: 900; margin-top: 6px; text-shadow: 0 0 20px rgba(56,189,248,0.7); }
    .metric-value-fin { color: #10b981; font-size: 26px; font-weight: 900; margin-top: 6px; text-shadow: 0 0 20px rgba(16,185,129,0.7); }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #fc3d21 0%, #990000 100%);
        color: white; border: 1px solid rgba(252,61,33,0.8); border-radius: 12px; font-weight: 900; padding: 0.85rem 1.8rem;
        box-shadow: 0 0 30px rgba(252, 61, 33, 0.7);
    }
    </style>
""", unsafe_allow_html=True)


# --- VERİ KATMANI ---
def utc_now():
    return datetime.now(timezone.utc)


def to_jd(t):
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)


def to_jd_arrays(times):
    # SatrecArray.sgp4 bitişik (C-contiguous) diziler ister; jf[:, 0] gibi sütun dilimleri hata verir
    jd, fr = zip(*(to_jd(t) for t in times))
    return np.array(jd), np.array(fr)


def embed_html(html, height):
    # st.components.v1.html yeni Streamlit sürümlerinde kaldırıldı -> st.iframe
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def geodetic_alt(r):
    n = np.linalg.norm(r, axis=-1)
    return n - RE * (1.0 - (r[..., 2] / n) ** 2 * WGS84_F)


def norad_id(l1):
    return l1[2:7].strip()


def parse_tles(text):
    # 3 satırlık blokları okur; bozuk/eksik bir blok gelirse kaymadan devam eder
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    out, i = [], 0
    while i + 2 < len(lines):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def elements_now(tles):
    if not tles:
        return np.array([]), np.array([])
    sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
    jd, fr = to_jd(utc_now())
    e, r, _ = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
    ok = e[:, 0] == 0
    return geodetic_alt(r[:, 0, :])[ok], np.degrees([s.inclo for s in sats])[ok]


GROUPS = ["active", "visual"]
# Gerçek enkaz grupları (CelesTrak). Hiçbiri alınamazsa eski davranışa (visual) dönülür.
DEBRIS_GROUPS = ["cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris", "cosmos-1408-debris"]


# İndirilen TLE'ler diske yazılır: CelesTrak aynı grubu 2 saat içinde tekrar indirene 403 döndürür,
# ayrıca sunucu yeniden başlatıldığında her şeyi baştan indirmek gerekmez.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tle_cache")
CACHE_TTL_S = 2 * 3600


def _fetch_group(g):
    """Dönüş: (tles, kaynak) — kaynak: 'live' | 'cache' | 'stale' | 'fail'"""
    path = os.path.join(CACHE_DIR, f"{g}.tle")
    has_cache = os.path.exists(path)
    if has_cache and time.time() - os.path.getmtime(path) < CACHE_TTL_S:
        with open(path, encoding="utf-8") as f:
            return parse_tles(f.read()), "cache"
    try:
        r = requests.get(
            f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle",
            timeout=(5, 30), headers=HEADERS
        )
        r.raise_for_status()
        tles = parse_tles(r.text)
        if not tles:
            raise ValueError("boş yanıt")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        return tles, "live"
    except Exception:
        # İndirme olmadıysa eski de olsa gerçek veriyi kullan
        if has_cache:
            with open(path, encoding="utf-8") as f:
                return parse_tles(f.read()), "stale"
        return [], "fail"


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_catalog():
    # Tüm gruplar paralel indirilir (sırayla indirmek her biri için ayrı bekleme demekti)
    all_groups = GROUPS + DEBRIS_GROUPS
    with ThreadPoolExecutor(max_workers=len(all_groups)) as ex:
        fetched = dict(zip(all_groups, ex.map(_fetch_group, all_groups)))

    sources = {g: src for g, (_, src) in fetched.items()}
    is_offline = all(src == "fail" for src in sources.values())

    # Alınamayan ana grup için önce diğer ana gruba, o da yoksa yedek TLE'lere düş
    data = {g: fetched[g][0] for g in GROUPS}
    for g in GROUPS:
        if not data[g]:
            data[g] = next((data[o] for o in GROUPS if data[o]), None) or parse_tles(FALLBACK_TLE)

    # Enkaz: gerçek debris grupları; olmazsa 'visual' grubuna düş
    debris, seen = [], set()
    for g in DEBRIS_GROUPS:
        for t in fetched[g][0]:
            nid = norad_id(t[1])
            if nid not in seen:
                seen.add(nid)
                debris.append(t)
    if not debris:
        debris = data.get("visual", [])

    a_alt, a_inc = elements_now(data.get("active", []))
    d_alt, d_inc = elements_now(debris)
    if is_offline:
        status_label = "OFFLINE 🟠 (Yedek Veri)"
    elif "fail" in sources.values():
        status_label = "PARTIAL 🟡 (Bazı gruplar alınamadı)"
    elif "stale" in sources.values():
        status_label = "CACHED 🟡 (Önbellek)"
    else:
        status_label = "LIVE 🟢"
    return {"groups": data, "debris": debris, "a_alt": a_alt, "a_inc": a_inc,
            "d_alt": d_alt, "d_inc": d_inc,
            "fetched": utc_now().strftime("%Y-%m-%d %H:%M:%S UTC"), "status_label": status_label}


def build_table(tles):
    if not tles:
        return pd.DataFrame()
    sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
    jd, fr = to_jd(utc_now())
    e, r, v = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
    rows = []
    for i, (name, l1, _) in enumerate(tles):
        if e[i, 0] != 0:
            continue
        s = sats[i]
        epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=(s.jdsatepoch + s.jdsatepochF - 2451545.0))
        alt = float(geodetic_alt(r[i, 0]))
        if name.upper().startswith(CREWED_PREFIXES):
            status = "CREWED 🟢"
        elif alt < 300:
            status = "DECAYING ⚠️"
        else:
            status = "NOMINAL"
        rows.append({
            "Satellite Name": name, "NORAD ID": norad_id(l1), "Epoch Time": epoch.strftime("%Y-%m-%d %H:%M:%S"),
            "Altitude (km)": round(alt), "Inclination (°)": round(math.degrees(s.inclo), 2),
            "Period (min)": round(2 * math.pi / s.no_kozai, 1),
            "Speed (km/s)": round(float(np.linalg.norm(v[i, 0])), 2),
            "Status": status,
        })
    return pd.DataFrame(rows)


CAT = load_catalog()


@st.cache_data(ttl=30, show_spinner=False)
def _table_cached(fetched, _tles):
    # fetched yalnızca önbellek anahtarı; tablo 30 sn boyunca yeniden hesaplanmaz
    return build_table(_tles)


df_api = _table_cached(CAT["fetched"], CAT["groups"]["visual"])


def _feed_label(s):
    if st.session_state.lang == "tr":
        return s
    return (s.replace("(Yedek Veri)", "(Fallback Data)")
             .replace("(Bazı gruplar alınamadı)", "(Some groups unavailable)")
             .replace("(Önbellek)", "(Cache)"))


FEED_LABEL = _feed_label(CAT["status_label"])

# Tablo başlıkları / değerleri Türkçe modunda yalnızca gösterimde çevrilir (veri aynı kalır)
_COLS_TR = {
    "Satellite Name": "Uydu Adı", "NORAD ID": "NORAD ID", "Epoch Time": "Epok Zamanı",
    "Altitude (km)": "İrtifa (km)", "Inclination (°)": "Eğim (°)", "Period (min)": "Periyot (dk)",
    "Speed (km/s)": "Hız (km/s)", "Status": "Durum",
    "Object A": "Nesne A", "Object B": "Nesne B", "Miss Distance (km)": "Geçiş Mesafesi (km)",
    "Rel. Velocity (km/s)": "Göreli Hız (km/s)", "TCA (UTC)": "TCA (UTC)", "Risk": "Risk",
}
_VALS_TR = {
    "CREWED 🟢": "MÜRETTEBATLI 🟢", "DECAYING ⚠️": "ALÇALIYOR ⚠️", "NOMINAL": "NORMAL",
    "🔴 CRITICAL": "🔴 KRİTİK", "🟠 HIGH": "🟠 YÜKSEK", "🟡 WATCH": "🟡 İZLEME", "⚪ SCREENED": "⚪ ELENDİ",
}


def _disp(df):
    if st.session_state.lang != "tr" or df is None or df.empty:
        return df
    d = df.copy()
    for c in ("Status", "Risk"):
        if c in d.columns:
            d[c] = d[c].replace(_VALS_TR)
    return d.rename(columns=_COLS_TR)

# --- SPLASH EKRANI ---
# Giriş ekranı: arka planda gerçek yörünge verisiyle dönen 3D Dünya + bilgilendirme bölümleri.
# Ana uygulamaya etkisi yoktur: CSS yalnızca bu çalıştırmada eklenir ve sonunda st.stop() çağrılır.

def _enter_app():
    st.session_state.intro_passed = True
    st.session_state.sg_just_entered = True


LIVE_URL = "https://www.youtube.com/watch?v=tj4knR4r1UU"
LIVE_THUMB = "https://i.ytimg.com/vi/tj4knR4r1UU/hqdefault.jpg"
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


def _live_set(on):
    st.session_state.sg_live_on = on


@_fragment
def _live_player_ui(play_lbl, cap):
    with _keyed("sg_livebox"):
        if st.session_state.get("sg_live_on", False):
            try:
                st.video(LIVE_URL, autoplay=True)
            except TypeError:
                st.video(LIVE_URL)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.button(T("✕ Oynatıcıyı kapat", "✕ Close player"), key="sg_live_close", on_click=_live_set, args=(False,))
            with c2:
                _yt_lbl = T("YouTube'da aç ↗", "Open on YouTube ↗")
                st.markdown(f'<a class="sg-ghost" style="height:42px" href="{LIVE_URL}" target="_blank" rel="noopener">{_yt_lbl}</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sg-poster"><img src="{LIVE_THUMB}" alt="ISS" loading="lazy"><span class="sg-poster-cap">{cap}</span></div>', unsafe_allow_html=True)
            st.button("▶  " + play_lbl, key="sg_live_open", on_click=_live_set, args=(True,), **stretch(st.button))


def _launch_from_url():
    # Navigasyondaki "Canlı İzlemeyi Başlat" bağlantısı ?launch=1 ile gelir
    try:
        return str(st.query_params.get("launch", "")) == "1"
    except Exception:
        return False


def _container_has_key():
    try:
        return "key" in inspect.signature(st.container).parameters
    except (TypeError, ValueError):
        return False


def _flat(html):
    # st.markdown girintili / boş satırlı HTML'i kod bloğu sanabilir; satırları tek satıra indir
    return " ".join(l.strip() for l in html.strip().splitlines() if l.strip())


def _fmt_int(n):
    return f"{n:,}".replace(",", ".")


def _sun_unit_three(jd_ut):
    """Güneşin yönü (Three.js eksenlerinde birim vektör). Düşük hassasiyetli güneş efemerisi (~0.01°)."""
    n = jd_ut - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    x, y, z = math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)
    return [round(x, 5), round(z, 5), round(-y, 5)]  # TEME (x, y, z) -> three.js (x, z, -y)


@st.cache_data(ttl=600, show_spinner=False)
def splash_globe_payload(n_active=1500, n_debris=1300):
    """Giriş ekranındaki küre için: gerçek TLE'lerden şu anki konum/hız (tek SGP4 adımı) + gerçek güneş yönü."""
    jd0, fr0 = to_jd(utc_now())
    out = {"kind": [], "p": [], "v": [], "sun": _sun_unit_three(jd0 + fr0),
           "gmst0": math.radians((280.46061837 + 360.98564736629 * (jd0 + fr0 - 2451545.0)) % 360),
           "omega": 7.2921159e-5}
    try:
        cat = load_catalog()
        rng = np.random.default_rng(21)

        def pick(lst, n):
            if not lst or n <= 0:
                return []
            return [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)]

        # İstasyonlar (ISS / CSS / Tiangong): her önekten bir tane, kendi yörünge halkasıyla çizilir
        crew, used = [], set()
        for t in cat["groups"]["visual"]:
            for p in CREWED_PREFIXES:
                if t[0].upper().startswith(p) and p not in used:
                    used.add(p)
                    crew.append(t)
        sel = ([(2, t) for t in crew] + [(0, t) for t in pick(cat["groups"]["active"], n_active)]
               + [(1, t) for t in pick(cat["debris"], n_debris)])
        seen, kinds, sats = set(), [], []
        for k, (name, l1, l2) in sel:
            nid = norad_id(l1)
            if nid in seen:
                continue
            seen.add(nid)
            kinds.append(k)
            sats.append(Satrec.twoline2rv(l1, l2))
        if not sats:
            return out
        e, r, v = SatrecArray(sats).sgp4(np.array([jd0]), np.array([fr0]))
        r, v = r[:, 0, :], v[:, 0, :]
        rn, vn = np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1)
        # Yalnızca LEO bandı ve o anda yaklaşık dairesel olanlar (animasyon dairesel yörünge yaklaşımı kullanır)
        ok = ((e[:, 0] == 0) & (rn > RE + 150) & (rn < RE + 2500)
              & (np.abs(np.einsum("ij,ij->i", r, v)) / (rn * vn) < 0.15))
        to3 = lambda a: np.stack([a[:, 0], a[:, 2], -a[:, 1]], axis=1)
        k_list = np.array(kinds)[ok].astype(int).tolist()
        p_list = np.round(to3(r[ok]) / RE, 4).ravel().tolist()
        v_list = np.round(to3(v[ok]), 3).ravel().tolist()
        out["kind"], out["p"], out["v"] = k_list, p_list, v_list
    except Exception:
        pass
    return out


INTRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --sg-cyan:#38bdf8; --sg-red:#fc3d21; --sg-green:#10b981; --sg-mute:#94a3b8; --sg-line:rgba(56,189,248,.16); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp { background:#040817; }
[data-testid="stMain"], section.main, .main { scroll-behavior:smooth; }
/* Canlı izlemeye geçerken eski giriş içeriği (soluk yazılar) ekranda kalmasın */
.st-key-sg_fg [data-stale="true"], .st-key-sg_fg [data-stale="true"] * { visibility:hidden !important; }
.stApp [data-testid="stAppViewContainer"] { overscroll-behavior:none; }
.sg-card, .sg-faq details, .sg-metric { contain:layout style; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:1240px !important; padding:0 clamp(18px,3vw,40px) 40px !important; }

/* Arka plan: tüm ekranı kaplayan sabit 3D Dünya */
.st-key-sg_bg { position:fixed !important; inset:0; width:100vw; height:100vh; z-index:0; pointer-events:none; margin:0; padding:0; }
.st-key-sg_bg > div, .st-key-sg_bg .element-container, .st-key-sg_bg [data-testid="stElementContainer"],
.st-key-sg_bg [data-testid="stIFrame"], .st-key-sg_bg iframe { width:100vw !important; height:100vh !important; max-width:none !important; margin:0 !important; border:0 !important; }
.st-key-sg_fg { position:relative; z-index:3; gap:.5rem; font-family:'Inter','Segoe UI',-apple-system,sans-serif; }

/* Sabit navigasyon */
.sg-nav { position:fixed; top:0; left:0; right:0; height:64px; z-index:50; display:flex; align-items:center; justify-content:space-between;
  padding:0 clamp(16px,3vw,40px); background:rgba(4,8,23,.88);
  border-bottom:1px solid var(--sg-line); font-family:'Inter','Segoe UI',sans-serif; }
.sg-logo { display:flex; align-items:center; gap:11px; color:#fff !important; text-decoration:none !important; font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:.16em; font-size:15px; }
.sg-logo-mark { width:22px; height:22px; border-radius:50%; border:1.5px solid var(--sg-cyan); position:relative; box-shadow:0 0 12px rgba(56,189,248,.4); }
.sg-logo-mark::after { content:""; position:absolute; width:5px; height:5px; border-radius:50%; background:var(--sg-red); top:-3px; right:1px; }
.sg-links { display:flex; gap:clamp(14px,2.2vw,34px); }
.sg-links a { color:var(--sg-mute) !important; text-decoration:none !important; font-size:13.5px; font-weight:500; transition:color .2s; }
.sg-links a:hover { color:#fff !important; }
.sg-cta { color:#fff !important; text-decoration:none !important; font-size:13px; font-weight:600; padding:9px 18px; border-radius:999px; white-space:nowrap;
  border:1px solid rgba(56,189,248,.55); background:rgba(56,189,248,.10); transition:all .2s; }
.sg-cta:hover { background:rgba(56,189,248,.22); border-color:var(--sg-cyan); }
.sg-veil { position:fixed; inset:0; z-index:-1; pointer-events:none;
  background:linear-gradient(90deg,rgba(4,8,23,.88) 0%,rgba(4,8,23,.60) 34%,rgba(4,8,23,.08) 62%,rgba(4,8,23,0) 100%),linear-gradient(0deg,rgba(4,8,23,.55),transparent 28%); }

/* Hero */
.sg-hero { padding:clamp(96px,17vh,170px) 0 22px; max-width:640px; animation:sgUp .9s ease both; font-family:'Inter','Segoe UI',sans-serif; }
.sg-badge { display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; border:1px solid var(--sg-line); background:rgba(8,16,38,.65);
  color:#bae6fd; font-size:11.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
.sg-dot { width:7px; height:7px; border-radius:50%; background:var(--sg-green); animation:sgPulse 2.2s infinite; }
.sg-dot.warn { background:#f59e0b; animation:none; }
.sg-h1 { font-family:'Space Grotesk','Inter',sans-serif; font-size:clamp(32px,4.1vw,58px); line-height:1.08; font-weight:700; letter-spacing:-.02em; color:#fff; margin:22px 0 18px; }
.sg-h1 em { font-style:normal; background:linear-gradient(90deg,#38bdf8,#e0f2fe); -webkit-background-clip:text; background-clip:text; color:transparent; }
.sg-lead { color:#cbd5e1; font-size:clamp(15px,1.25vw,17.5px); line-height:1.7; max-width:560px; margin:0; }
[class*="st-key-sg_launch"] button { height:54px; border-radius:999px !important; border:1px solid rgba(252,61,33,.9) !important;
  background:linear-gradient(135deg,#fc3d21 0%,#c42a12 100%) !important; padding:0 26px !important;
  box-shadow:0 10px 34px rgba(252,61,33,.32), inset 0 1px 0 rgba(255,255,255,.18) !important; transition:transform .2s ease, box-shadow .2s ease !important; }
[class*="st-key-sg_launch"] button:hover { transform:translateY(-2px); box-shadow:0 16px 44px rgba(252,61,33,.48), inset 0 1px 0 rgba(255,255,255,.22) !important; }
[class*="st-key-sg_launch"] button p { color:#fff !important; font-weight:700 !important; font-size:15px !important; letter-spacing:.01em; }
.sg-ghost { display:flex; align-items:center; justify-content:center; height:54px; border-radius:999px; border:1px solid rgba(148,163,184,.35);
  color:#e2e8f0 !important; text-decoration:none !important; font-weight:600; font-size:14.5px; transition:all .2s; font-family:'Inter','Segoe UI',sans-serif; }
.sg-ghost:hover { border-color:var(--sg-cyan); color:#fff !important; background:rgba(56,189,248,.08); }
.sg-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin-top:34px; max-width:640px; border:1px solid var(--sg-line);
  border-radius:16px; overflow:hidden; background:var(--sg-line); font-family:'Inter','Segoe UI',sans-serif; }
.sg-metric { background:rgba(6,12,30,.80); padding:14px 16px; }
.sg-metric b { display:block; font-family:'Space Grotesk','Inter',sans-serif; font-size:21px; color:#fff; font-weight:700; }
.sg-metric b.ok { color:var(--sg-green); } .sg-metric b.warn { color:#f59e0b; }
.sg-metric span { font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:var(--sg-mute); }
.sg-src { margin-top:16px; color:#64748b; font-size:11px; letter-spacing:.12em; text-transform:uppercase; font-family:'Inter','Segoe UI',sans-serif; }
.sg-src i { font-style:normal; color:#cbd5e1; margin-left:12px; padding:3px 10px; border:1px solid var(--sg-line); border-radius:999px; letter-spacing:.04em; text-transform:none; font-size:11.5px; }

/* Bölümler */
.sg-sec { margin-top:clamp(80px,12vh,140px); scroll-margin-top:90px; font-family:'Inter','Segoe UI',sans-serif; }
.st-key-sg_livebox { max-width:900px; margin:26px auto 0; }
.sg-poster { position:relative; border-radius:18px; overflow:hidden; border:1px solid var(--sg-line); aspect-ratio:16/9; background:#060b1a; }
.sg-poster img { width:100%; height:100%; object-fit:cover; display:block; opacity:.85; }
.sg-poster-cap { position:absolute; left:14px; bottom:10px; color:#cbd5e1; font-size:12px; text-shadow:0 1px 4px #000; }
.st-key-sg_livebox .stButton > button { height:48px; border-radius:999px; border:1px solid rgba(252,61,33,.9); background:linear-gradient(135deg,#fc3d21,#c42a12); margin-top:10px; }
.st-key-sg_livebox .stButton > button p { color:#fff; font-weight:700; }
.st-key-sg_livebox [data-testid="stVideo"], .st-key-sg_livebox iframe { border-radius:18px; overflow:hidden; }
.sg-sec.sg-first { margin-top:clamp(56px,8vh,90px); }
.sg-kicker { color:var(--sg-cyan); font-size:12px; letter-spacing:.22em; font-weight:700; text-transform:uppercase; }
.sg-h2 { font-family:'Space Grotesk','Inter',sans-serif; font-size:clamp(26px,3vw,40px); color:#fff; font-weight:700; letter-spacing:-.015em; line-height:1.15; margin:10px 0 14px; max-width:780px; }
.sg-p { color:#a9b8cc; font-size:15.5px; line-height:1.75; max-width:780px; margin:0 0 8px; }
.sg-p b { color:#fff; }
.sg-muted { color:#7f90a8; font-size:14px; }
.sg-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; margin-top:30px; }
.sg-card { padding:24px; border-radius:18px; background:linear-gradient(160deg,rgba(14,24,48,.84),rgba(6,11,26,.92)); border:1px solid var(--sg-line);
  transition:border-color .25s ease; }
.sg-card:hover { border-color:rgba(56,189,248,.45); }
.sg-stat { font-family:'Space Grotesk','Inter',sans-serif; font-size:34px; font-weight:700; color:var(--sg-cyan); letter-spacing:-.02em; }
.sg-huge { font-size:clamp(44px,6vw,76px); }
.sg-ct { color:#fff; font-size:16px; font-weight:600; margin:8px 0 8px; }
.sg-cp { color:#94a3b8; font-size:14px; line-height:1.65; margin:0; }
.sg-step-n { font-family:'Space Grotesk','Inter',sans-serif; color:rgba(56,189,248,.6); font-size:13px; letter-spacing:.2em; font-weight:700; }
.sg-price { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; margin-top:30px; }
.sg-tier { position:relative; display:flex; flex-direction:column; gap:16px; border-top:2px solid var(--c,#38bdf8); }
.sg-tier.pop { border-color:rgba(56,189,248,.55); background:linear-gradient(160deg,rgba(10,36,58,.78),rgba(6,11,26,.94)); box-shadow:0 20px 60px rgba(6,182,212,.12); }
.sg-tag { position:absolute; top:18px; right:18px; background:#06b6d4; color:#04121f; font-size:10px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; padding:4px 10px; border-radius:999px; }
.sg-tier-name { font-family:'Space Grotesk','Inter',sans-serif; font-size:21px; color:#fff; font-weight:700; }
.sg-lbl { font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--c,#38bdf8); font-weight:700; margin-bottom:5px; }
.sg-val { color:#cbd5e1; font-size:14px; line-height:1.65; }
.sg-faq { margin-top:60px; max-width:900px; }
.sg-faq details { border:1px solid var(--sg-line); border-radius:14px; background:rgba(8,14,32,.80); margin-bottom:10px; padding:0 22px; transition:border-color .2s; }
.sg-faq details[open] { border-color:rgba(56,189,248,.4); }
.sg-faq summary { cursor:pointer; padding:18px 0; color:#fff; font-weight:600; font-size:15.5px; list-style:none; display:flex; justify-content:space-between; gap:16px; }
.sg-faq summary::-webkit-details-marker { display:none; }
.sg-faq summary::after { content:"+"; color:var(--sg-cyan); font-size:22px; line-height:1; }
.sg-faq details[open] summary::after { content:"–"; }
.sg-faq details p { color:#a9b8cc; font-size:14.5px; line-height:1.75; padding:0 0 18px; margin:0; }
.sg-about { display:grid; grid-template-columns:1.3fr 1fr; gap:36px; align-items:center; }
.sg-final { text-align:center; }
.sg-final .sg-h2, .sg-final .sg-p { margin-left:auto; margin-right:auto; }
.sg-foot { margin-top:70px; padding-top:20px; border-top:1px solid var(--sg-line); color:#64748b; font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; text-align:center; font-family:'Inter','Segoe UI',sans-serif; }
.sg-top { position:fixed; right:24px; bottom:54px; z-index:60; display:flex; align-items:center; gap:8px; padding:11px 18px; border-radius:999px;
  background:rgba(6,12,30,.82); border:1px solid rgba(56,189,248,.45); color:#e0f2fe !important; text-decoration:none !important;
  font-family:'Inter','Segoe UI',sans-serif; font-size:13px; font-weight:600;
  box-shadow:0 4px 14px rgba(0,0,0,.45); transition:all .2s ease; }
.sg-top:hover { border-color:var(--sg-cyan); background:rgba(56,189,248,.18); transform:translateY(-2px); }

@keyframes sgUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
@keyframes sgPulse { 0% { box-shadow:0 0 0 0 rgba(16,185,129,.55); } 70% { box-shadow:0 0 0 9px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@media (max-width:900px) {
  .sg-links { display:none; }
  .sg-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .sg-about { grid-template-columns:1fr; }
  .sg-veil { background:rgba(4,8,23,.62); }
}
@media (prefers-reduced-motion:reduce) { .sg-hero, .sg-dot { animation:none; } }
</style>
"""

# Arka plan küresi (Three.js). Yer işareti: __DATA__ -> splash_globe_payload() çıktısı (JSON)
INTRO_GLOBE_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html, body { margin:0; height:100%; overflow:hidden; background:radial-gradient(ellipse at 72% 45%, #0b1329 0%, #040817 55%, #000000 100%); }
  #c { position:fixed; inset:0; opacity:0; transition:opacity 1.8s ease; }
  body.ready #c { opacity:1; }
  #cap { position:fixed; right:22px; bottom:16px; font:500 10.5px/1.4 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.14em; text-transform:uppercase; color:rgba(148,163,184,.7); }
  #cap b { color:#38bdf8; font-weight:600; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="c"></div><div id="cap"></div>
<script>
setTimeout(function () { document.body.classList.add('ready'); }, 3000);
(function () {
  try {
    const D = __DATA__;
    const RE = 6378.137, TAU = Math.PI * 2;
    const RATE = (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) ? 12 : 80;   // simülasyon hızı (gerçek zamanın katı)
    const BASE = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/';
    const PR = Math.min(window.devicePixelRatio || 1, 1.25);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(PR);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('c').appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 600);
    // Gerçek güneş yönü (inertial); kamera güneşten ~54° yana yerleşir -> aydınlık yarım küre + gün/gece sınırı görünür
    const sunDir = new THREE.Vector3(D.sun[0], D.sun[1], D.sun[2]).normalize();
    const camDir = sunDir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.95);
    camDir.y = 0.30; camDir.normalize();

    function flat(r, g, b) {
      const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1, THREE.RGBAFormat);
      t.needsUpdate = true; return t;
    }
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');
    const aniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    function load(name, done) { loader.load(BASE + name, function (t) { t.anisotropy = aniso; done(t); }, undefined, function () {}); }

    // ---- Dünya: gün / gece (şehir ışıkları) / okyanus parıltısı / atmosfer kenarı ----
    const earthMat = new THREE.ShaderMaterial({
      uniforms: { dayMap: { value: flat(8, 38, 92) }, nightMap: { value: flat(0, 0, 0) }, sunDir: { value: sunDir } },
      vertexShader: `
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          vUv = uv;
          vN = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vV = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: `
        uniform sampler2D dayMap; uniform sampler2D nightMap; uniform vec3 sunDir;
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          vec3 N = normalize(vN); vec3 V = normalize(vV);
          float d = dot(N, sunDir);
          float lit = smoothstep(-0.10, 0.24, d);
          vec3 day = texture2D(dayMap, vUv).rgb;
          vec3 city = texture2D(nightMap, vUv).rgb;
          vec3 dayCol = day * (0.16 + 0.96 * clamp(d, 0.0, 1.0));
          vec3 nightCol = day * 0.035 + city * vec3(1.0, 0.80, 0.52) * 1.5;
          vec3 col = mix(nightCol, dayCol, lit);
          float ocean = clamp((day.b - day.r - 0.04) * 4.0, 0.0, 1.0);
          vec3 R = reflect(-sunDir, N);
          col += vec3(0.75, 0.88, 1.0) * pow(max(dot(R, V), 0.0), 60.0) * ocean * lit * 0.35;
          float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
          col += vec3(0.20, 0.52, 1.0) * rim * (0.20 + 0.80 * lit) * 0.75;
          float tw = exp(-pow((d - 0.03) / 0.10, 2.0));
          col += vec3(1.0, 0.45, 0.18) * tw * rim * 0.9;
          gl_FragColor = vec4(col, 1.0);
        }`
    });
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), earthMat);

    const cloudMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: flat(0, 0, 0) }, sunDir: { value: sunDir } },
      transparent: true, depthWrite: false,
      vertexShader: `
        varying vec2 vUv; varying vec3 vN;
        void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform sampler2D map; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN;
        void main() {
          vec4 t = texture2D(map, vUv);
          float c = t.r * t.a;
          float lit = smoothstep(-0.12, 0.30, dot(normalize(vN), sunDir));
          gl_FragColor = vec4(vec3(0.94, 0.97, 1.0) * (0.10 + 0.90 * lit), c * 0.5 * (0.2 + 0.8 * lit));
        }`
    });
    const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.007, 96, 64), cloudMat);

    const atmoMat = new THREE.ShaderMaterial({
      uniforms: { sunDir: { value: sunDir } },
      transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
      vertexShader: `
        varying vec3 vN; varying vec3 vWN;
        void main() { vN = normalize(normalMatrix * normal); vWN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform vec3 sunDir; varying vec3 vN; varying vec3 vWN;
        void main() {
          float d = clamp(-dot(normalize(vN), vec3(0.0, 0.0, 1.0)), 0.0, 1.0);
          float i = pow(clamp(d / 0.5, 0.0, 1.0), 2.4);
          float k = 0.45 + 0.55 * smoothstep(-0.40, 0.45, dot(normalize(vWN), sunDir));
          gl_FragColor = vec4(vec3(0.22, 0.58, 1.0) * i * k * 0.95, 1.0);
        }`
    });
    const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.16, 96, 64), atmoMat);
    scene.add(earth, clouds, atmo);

    let ready = false;
    function markReady() { if (!ready) { ready = true; document.body.classList.add('ready'); } }
    load('earth_atmos_2048.jpg', function (t) { earthMat.uniforms.dayMap.value = t; markReady(); });
    load('earth_lights_2048.png', function (t) { earthMat.uniforms.nightMap.value = t; });
    load('earth_clouds_1024.png', function (t) { cloudMat.uniforms.map.value = t; });

    // ---- Yıldızlar ----
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,.85)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
    const DOT = new THREE.CanvasTexture(cv);

    function stars(n, size, opacity) {
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const u = Math.random() * 2 - 1, th = Math.random() * TAU, s = Math.sqrt(1 - u * u), R = 200;
        pos[3*i] = R * s * Math.cos(th); pos[3*i+1] = R * u; pos[3*i+2] = R * s * Math.sin(th);
        const k = Math.random(), c = k < 0.15 ? [1, 0.85, 0.7] : (k < 0.4 ? [0.7, 0.82, 1] : [0.9, 0.93, 1]), b = 0.45 + 0.55 * Math.random();
        col[3*i] = c[0] * b; col[3*i+1] = c[1] * b; col[3*i+2] = c[2] * b;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ size: size * PR, sizeAttenuation: false, map: DOT, vertexColors: true, transparent: true, opacity: opacity, depthWrite: false }));
      p.frustumCulled = false; return p;
    }
    const starGroup = new THREE.Group();
    starGroup.add(stars(2400, 2.6, 0.85), stars(140, 5, 1.0));
    scene.add(starGroup);

    // ---- Gerçek yörünge nesneleri: şu anki gerçek konum/hızdan dairesel yörünge yaklaşımı ----
    const N = D.kind.length;
    const P0 = new Float32Array(N * 3), Q0 = new Float32Array(N * 3), W = new Float32Array(N);
    const mainIdx = [], crewIdx = [];
    for (let i = 0; i < N; i++) {
      const x = D.p[3*i], y = D.p[3*i+1], z = D.p[3*i+2];
      const vx = D.v[3*i] / RE, vy = D.v[3*i+1] / RE, vz = D.v[3*i+2] / RE;
      const rr = Math.hypot(x, y, z), vv = Math.hypot(vx, vy, vz) || 1e-9, w = vv / rr;
      W[i] = w;
      P0[3*i] = x; P0[3*i+1] = y; P0[3*i+2] = z;
      Q0[3*i] = vx / w; Q0[3*i+1] = vy / w; Q0[3*i+2] = vz / w;
      (D.kind[i] === 2 ? crewIdx : mainIdx).push(i);
    }
    function makePoints(idx, hex, size, opacityVal, perKind) {
      const g = new THREE.BufferGeometry();
      const pos = new Float32Array(idx.length * 3), col = new Float32Array(idx.length * 3);
      const cA = new THREE.Color(hex), cD = new THREE.Color(0xfc3d21);
      idx.forEach(function (i, j) {
        const c = (perKind && D.kind[i] === 1) ? cD : cA;
        col[3*j] = c.r; col[3*j+1] = c.g; col[3*j+2] = c.b;
      });
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ size: size * PR, sizeAttenuation: false, map: DOT, vertexColors: true, transparent: true,
        opacity: opacityVal, depthWrite: false, blending: THREE.AdditiveBlending }));
      p.frustumCulled = false; scene.add(p);
      return { pts: p, pos: pos, geo: g, idx: idx };
    }
    const mainSet = makePoints(mainIdx, 0x38bdf8, 3.4, 0.95, true);     // mavi: aktif uydu, kırmızı: enkaz
    const crewSet = makePoints(crewIdx, 0x10b981, 8, 1.0, false);        // yeşil: insanlı istasyonlar
    crewIdx.forEach(function (i) {
      const pts = [];
      for (let k = 0; k < 160; k++) {
        const a = TAU * k / 160, c = Math.cos(a), s = Math.sin(a);
        pts.push(new THREE.Vector3(P0[3*i] * c + Q0[3*i] * s, P0[3*i+1] * c + Q0[3*i+1] * s, P0[3*i+2] * c + Q0[3*i+2] * s));
      }
      scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.35 })));
    });
    function place(set, t) {
      const idx = set.idx, pos = set.pos;
      for (let j = 0; j < idx.length; j++) {
        const i = idx[j], a = W[i] * t, c = Math.cos(a), s = Math.sin(a);
        pos[3*j]   = P0[3*i]   * c + Q0[3*i]   * s;
        pos[3*j+1] = P0[3*i+1] * c + Q0[3*i+1] * s;
        pos[3*j+2] = P0[3*i+2] * c + Q0[3*i+2] * s;
      }
      set.geo.attributes.position.needsUpdate = true;
    }

    // ---- Boyut / kamera: geniş ekranda küre sağa kayar, metin solda kalır ----
    function resize() {
      const w = Math.max(window.innerWidth, 1), h = Math.max(window.innerHeight, 1), asp = w / h;
      renderer.setSize(w, h);
      camera.aspect = asp;
      const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const wide = asp > 1.15;
      const dist = wide ? 3.8 : Math.max(3.8, 1.35 / (half * asp));
      camera.position.copy(camDir).multiplyScalar(dist);
      camera.lookAt(0, 0, 0);
      if (wide) camera.setViewOffset(w, h, -w * 0.20, 0, w, h); else camera.clearViewOffset();
      camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener('resize', resize);
    setTimeout(resize, 400);

    if (N > 0) {
      document.getElementById('cap').innerHTML = '<b>Gerçek TLE verisi</b> · SGP4 · ' + N + ' nesne · ' + RATE + '× zaman';
    }

    let t = 0, last = performance.now();
    function frame(now) {
      requestAnimationFrame(frame);
      if (now - last < 32) return;
      const dt = Math.min((now - last) / 1000, 0.1); last = now;
      t += dt * RATE;
      earth.rotation.y = D.gmst0 + D.omega * t;          // Greenwich meridyeni gerçek GMST'ye göre
      clouds.rotation.y = earth.rotation.y + t * 0.00003;
      starGroup.rotation.y = t * 0.00003;
      place(mainSet, t); place(crewSet, t);
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  } catch (err) { console.error(err); }
})();
</script></body></html>
"""


# ===== Uzay çöpü tarihçesi (giriş sayfasının sonu) =====
# (yıl, toplam, aktif uydu, uzay çöpü, çöp oranı %) · ESA MASTER (2026) ve NASA / USSTRATCOM derlemesi · yaklaşık değerler
DEBRIS_HISTORY = [
    (1960, 200, 15, 185, 92.5),
    (1970, 1500, 150, 1350, 90.0),
    (1980, 4200, 350, 3850, 91.7),
    (1990, 5600, 450, 5150, 92.0),
    (2000, 7800, 650, 7150, 91.7),
    (2009, 18000, 950, 17050, 94.7),
    (2016, 19200, 1450, 17750, 92.4),
    (2020, 24500, 3300, 21200, 86.5),
    (2022, 30500, 6700, 23800, 78.0),
    (2024, 35000, 9800, 25200, 72.0),
    (2025, 44964, 11300, 33664, 74.9),
    (2026, 46650, 11300, 35350, 75.8),
]


def _dh_num(v, tr):
    s = f"{v:,}"
    return s.replace(",", ".") if tr else s


def _dh_pct(r, tr):
    return f"%{r:.1f}".replace(".", ",") if tr else f"{r:.1f}%"


def _debris_bars_svg(tr):
    W, H, pl, pr, pt, pb = 720, 300, 62, 12, 34, 34
    cw, ch = W - pl - pr, H - pt - pb
    vmax = 50000
    base = pt + ch
    slot = cw / len(DEBRIS_HISTORY)
    bw = slot * 0.6
    t_act = "Aktif uydu" if tr else "Active satellites"
    t_deb = "Uzay çöpü" if tr else "Debris"
    out = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block" role="img">']
    out.append(f'<rect x="{pl}" y="8" width="12" height="12" rx="2" fill="#06b6d4"/>')
    out.append(f'<text x="{pl + 18}" y="18" fill="#cbd5e1" font-size="12">{t_act}</text>')
    out.append(f'<rect x="{pl + 130}" y="8" width="12" height="12" rx="2" fill="#f97316"/>')
    out.append(f'<text x="{pl + 148}" y="18" fill="#cbd5e1" font-size="12">{t_deb}</text>')
    for v in range(0, vmax + 1, 10000):
        yy = base - ch * v / vmax
        out.append(f'<line x1="{pl}" x2="{W - pr}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="rgba(148,163,184,.16)"/>')
        out.append(f'<text x="{pl - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="#64748b" font-size="11">{_dh_num(v, tr)}</text>')
    for i, (yr, _tot, act, deb, _pct) in enumerate(DEBRIS_HISTORY):
        x = pl + slot * i + (slot - bw) / 2
        h_act = ch * act / vmax
        h_deb = ch * deb / vmax
        out.append(f'<rect x="{x:.1f}" y="{base - h_act:.1f}" width="{bw:.1f}" height="{h_act:.1f}" fill="#06b6d4"/>')
        out.append(f'<rect x="{x:.1f}" y="{base - h_act - h_deb:.1f}" width="{bw:.1f}" height="{h_deb:.1f}" fill="#f97316" opacity=".85"/>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{H - 10}" text-anchor="middle" fill="#94a3b8" font-size="11">{yr}</text>')
    out.append("</svg>")
    return "".join(out)


def _debris_table_html(tr):
    heads = (["Yıl", "Toplam", "Aktif uydu", "Uzay çöpü", "Çöp oranı"] if tr
             else ["Year", "Total", "Active satellites", "Debris", "Debris share"])
    th = ("padding:8px 10px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;"
          "color:#94a3b8;text-align:right;white-space:nowrap;border-bottom:1px solid rgba(56,189,248,.25)")
    td = ("padding:7px 10px;text-align:right;color:#cbd5e1;font-size:13.5px;white-space:nowrap;"
          "font-variant-numeric:tabular-nums;border-bottom:1px solid rgba(56,189,248,.08)")
    head_cells = []
    for k, h in enumerate(heads):
        align = "left" if k == 0 else "right"
        head_cells.append(f'<th style="{th};text-align:{align}">{h}</th>')
    body = []
    for yr, tot, act, deb, pct in DEBRIS_HISTORY:
        peak = yr == 2009
        row_bg = "background:rgba(249,115,22,.08);" if peak else ""
        pct_css = "color:#fb923c;font-weight:700;" if peak else ""
        body.append(
            f'<tr style="{row_bg}">'
            f'<td style="{td};text-align:left;color:#fff;font-weight:600">{yr}</td>'
            f'<td style="{td}">{_dh_num(tot, tr)}</td>'
            f'<td style="{td}">{_dh_num(act, tr)}</td>'
            f'<td style="{td}">{_dh_num(deb, tr)}</td>'
            f'<td style="{td};{pct_css}">{_dh_pct(pct, tr)}</td>'
            "</tr>"
        )
    head_html = "".join(head_cells)
    body_html = "".join(body)
    return ('<table style="width:100%;border-collapse:collapse;min-width:520px">'
            f'<thead><tr>{head_html}</tr></thead>'
            f'<tbody>{body_html}</tbody></table>')


def _debris_cards_html(tr):
    if tr:
        items = [
            ("1960", "Uzay çağı başlıyor", "İlk fırlatmalar ve roket gövdeleri; yörüngedeki nesne sayısı yaklaşık 200."),
            ("2007", "Çin ASAT testi", "Fengyun-1C uydusunun bilinçli olarak vurulması binlerce enkaz parçası oluşturdu."),
            ("2009", "İlk kazara çarpışma", "10 Şubat 2009'da 790 km irtifada aktif Iridium 33 ile işlevsiz Cosmos 2251 çarpıştı; 1.800'den fazla yeni enkaz izlendi."),
            ("2020", "Mega takımyıldızlar", "2019 sonrasındaki Starlink benzeri yoğun fırlatımlarla aktif uydu sayısı 1.500'lerden 11.300'ün üzerine çıktı."),
        ]
    else:
        items = [
            ("1960", "Space age begins", "First launches and rocket bodies; about 200 objects in orbit."),
            ("2007", "Chinese ASAT test", "The deliberate destruction of Fengyun-1C created thousands of debris fragments."),
            ("2009", "First accidental collision", "On 10 February 2009 at 790 km, active Iridium 33 and defunct Cosmos 2251 collided; over 1,800 new debris pieces were tracked."),
            ("2020", "Mega-constellations", "After 2019, Starlink-style launches raised active satellites from about 1,500 to over 11,300."),
        ]
    cards = [
        f'<div class="sg-card"><div class="sg-stat">{yr}</div><div class="sg-ct">{ttl}</div><p class="sg-cp">{txt}</p></div>'
        for yr, ttl, txt in items
    ]
    return "".join(cards)


def _debris_history_html():
    tr = st.session_state.lang == "tr"
    if tr:
        kicker = "Veri analizi"
        title = "Uzay çöplerinin tarihçesi ve grafikleri"
        lead = ("1960'ta yörüngede yaklaşık 200 nesne vardı; bugün bu sayı yaklaşık 46.650. "
                "Uzay çöpü oranı 2009'da %94,7 ile zirveye ulaştı; aktif uydu sayısındaki hızlı artışla %75,8'e geriledi. "
                "Oran düşse de uzay çöpü sayısı 2009'dan bu yana iki katından fazla arttı.")
        cap_chart = "Yörüngedeki nesneler: aktif uydu ve uzay çöpü (yaklaşık)"
        cap_table = "Yıllara göre sayılar"
        note = ("Değerler kaynağa göre yaklaşıktır ve 10 cm'den büyük, kataloglanmış nesneleri yansıtır. "
                "Vurgulanan satır çöp oranının zirvesidir.")
        src_lbl = "Kaynaklar"
    else:
        kicker = "Data analysis"
        title = "History of space debris and charts"
        lead = ("In 1960 there were about 200 cataloged objects in orbit; today the count is roughly 46,650. "
                "The debris share peaked at 94.7% in 2009 and fell to 75.8% as active satellites surged. "
                "Even so, the number of debris objects has more than doubled since 2009.")
        cap_chart = "Objects in orbit: active satellites and debris (approximate)"
        cap_table = "Counts by year"
        note = ("Values are approximate, per the source, and reflect cataloged objects larger than 10 cm. "
                "The highlighted row marks the peak debris share.")
        src_lbl = "Sources"
    return f"""
    <div id="history" class="sg-sec">
      <div class="sg-kicker">{kicker}</div>
      <div class="sg-h2">{title}</div>
      <p class="sg-p">{lead}</p>
      <div style="margin-top:28px;padding:22px 22px 14px;border-radius:18px;border:1px solid var(--sg-line);background:linear-gradient(160deg,rgba(14,24,48,.84),rgba(6,11,26,.92))">
        <div class="sg-lbl">{cap_chart}</div>
        {_debris_bars_svg(tr)}
      </div>
      <div style="margin-top:16px;padding:18px 18px 12px;border-radius:18px;border:1px solid var(--sg-line);background:rgba(8,14,32,.80)">
        <div class="sg-lbl">{cap_table}</div>
        <div style="overflow-x:auto">{_debris_table_html(tr)}</div>
        <p class="sg-cp" style="margin-top:10px;font-size:12px">{note}</p>
      </div>
      <div class="sg-grid">{_debris_cards_html(tr)}</div>
      <div class="sg-src">{src_lbl} <i>ESA MASTER (2026)</i><i>NASA / USSTRATCOM</i><i>NASA (2009)</i></div>
    </div>
    """


def render_intro():
    n_active = len(CAT["groups"]["active"])
    n_total = n_active + len(CAT["debris"])
    feed_word = FEED_LABEL.split(" ")[0]
    keyed = _container_has_key()

    # Küre: tüm ekranı kaplayan sabit arka plan (container(key=...) desteklenmiyorsa atlanır)
    if keyed:
        with st.container(key="sg_bg"):
            embed_html(INTRO_GLOBE_HTML.replace("__DATA__", json.dumps(splash_globe_payload())), height=800)

    nav_html = """
    <div class="sg-nav">
      <a class="sg-logo" href="#top"><span class="sg-logo-mark"></span>SPACE GRAVE</a>
      <div class="sg-links"><a href="#features">Özellikler</a><a href="#pricing">B2B Fiyatlandırma</a><a href="#docs">Şartname / Dokümantasyon</a><a href="#about">Hakkımızda</a></div>
      <a class="sg-cta" href="?launch=1&amp;lang=tr" target="_self">Canlı İzlemeyi Başlat</a>
    </div>
    <div class="sg-veil"></div>
    <a class="sg-top" href="#top" aria-label="Ana ekrana dön">↑ Ana ekran</a>
    """

    hero_html = f"""
    <div id="top" class="sg-hero">
      <div class="sg-badge"><span class="sg-dot{'' if feed_word == 'LIVE' else ' warn'}"></span>B2B Uzay Karar Destek Platformu</div>
      <div class="sg-h1">Space Grave ile Alçak Dünya Yörüngesi (LEO) için <em>akıllı çarpışma önleme</em> ve karar destek platformu</div>
      <p class="sg-lead">Space Grave karar destek sistemimiz ile 50 km güvenlik küresi içindeki uydu filolarınızı uzay çöplerinden koruyun ve katastrofik yıkım risklerini gerçekleşmeden öngörün.</p>
    </div>
    """

    metrics_html = f"""
    <div class="sg-metrics">
      <div class="sg-metric"><b>{_fmt_int(n_total)}</b><span>Takip edilen nesne</span></div>
      <div class="sg-metric"><b>50 km</b><span>Güvenlik küresi</span></div>
      <div class="sg-metric"><b>40 J/g</b><span>NASA parçalanma eşiği</span></div>
      <div class="sg-metric"><b class="{'ok' if feed_word == 'LIVE' else 'warn'}">{feed_word}</b><span>CelesTrak veri akışı</span></div>
    </div>
    <div class="sg-src">Güvenilir veri kaynakları <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
    """

    sections_html = """
    <div id="threat" class="sg-sec sg-first">
      <div class="sg-kicker">Uzay Çöpleri</div>
      <div class="sg-h2">Yörüngedeki Sessiz Tehdit: Uzay Çöpleri</div>
      <p class="sg-p">İlk uydu fırlatmalarıyla başlayan süreçte, görev ömrü biten araçlar ve üst roket kademeleri yörüngede bırakıldı; kataloglanan nesne sayısı 1990 itibarıyla 5.600 seviyesine ulaştı. 10 Şubat 2009 tarihinde ise bir dönüm noktası yaşandı. Iridium 33 &amp; Cosmos 2251 790 kilometre irtifada çarpıştı. Bu vaka, tarihteki ilk kasıtsız hiper hızlı uydu çarpışması olarak kayıtlara geçti. Tek bir olayda takip edilemeyen 200.000 civarında küçük parça ve 1.800'den fazla büyük parça yörüngeye saçıldı. Yörüngede başıboş dolaşan 6.000'den fazla işlevsiz uydu ve roket gövdeleri, aktif roldeki ticari filolar ve sigorta portföyleri için her yıl zincirleme çarpışma riskini artırıyor.</p>
    </div>

    <div id="features" class="sg-sec">
      <div class="sg-kicker">Özellikler</div>
      <div class="sg-h2">Uzay kirliliği artık yalnızca teknik değil, finansal bir risk</div>
      <p class="sg-p">Uzay kirliliği, Alçak Dünya Yörüngesi'nde (LEO) çalışan uydu filoları için sadece teknik bir tehdit değil, aynı zamanda <b>&#36;42.3B</b> dolarlık "gizli bir finansal yüktür." Space Grave, karmaşık yörünge verilerini işleyerek uydu operatörleri ve uzay sigortacıları için anlık risk öngörüsü ve maliyet optimizasyonu sunar.</p>
      <div class="sg-grid">
        <div class="sg-card"><div class="sg-stat">50 km</div><div class="sg-ct">Güvenlik Küresi</div><p class="sg-cp">Yüksek riskli yakınlaşmalarda otomatik Kırmızı Alarm tetikler.</p></div>
        <div class="sg-card"><div class="sg-stat">SGP4</div><div class="sg-ct">Astrodinamik Motor ve 3D Görselleştirici</div><p class="sg-cp">Canlı TLE verilerinden hesaplanan koordinatlar 3D canlı Dünya görselleştiricisinde izlenir; mesafe tarayıcı yakınlaşmaları tespit eder.</p></div>
        <div class="sg-card"><div class="sg-stat">40 J/g</div><div class="sg-ct">NASA Parçalanma Standardı</div><p class="sg-cp">Kinetik enerji hesaplamalarıyla yıkım potansiyelini öngörür.</p></div>
        <div class="sg-card"><div class="sg-stat">0–100</div><div class="sg-ct">Sigorta Risk Skoru</div><p class="sg-cp">Roket gövdeleri ve yüksek yoğunluklu çöpler için dinamik poliçe analizi sunar.</p></div>
      </div>
      <p class="sg-p sg-muted" style="margin-top:28px">&#36;42.3B'lık uzay çöpü tehdidine karşı canlı TLE verilerini SGP4 motoru ve NASA parçalanma standartlarıyla işleyerek 50 km güvenlik küresi içindeki uzay çöplerini tarar; 0-100 arası dinamik sigorta risk skorunu ve kaçınma maliyetlerini anlık hesaplayarak manevra yakıt maliyetlerinizi optimize eder, hizmet kesintilerini ve zararlarınızı minimuma indiririz.</p>
    </div>

    <div id="docs" class="sg-sec">
      <div class="sg-kicker">Şartname / Dokümantasyon</div>
      <div class="sg-h2">Modüler ve ölçeklenebilir sistem mimarisi</div>
      <p class="sg-p">Platformumuz, canlı verinin çekilmesinden 3D arayüzde sunulmasına kadar 4 ana katmandan oluşan yüksek performanslı bir altyapıya sahiptir:</p>
      <div class="sg-grid">
        <div class="sg-card"><div class="sg-step-n">01</div><div class="sg-ct">Veri Toplama Katmanı</div><p class="sg-cp"><b style="color:#cbd5e1">Data Ingestion</b> — CelesTrak, Space-Track ve ESA DISCOS API'lerinden canlı TLE yörünge verileri ve uzay çöplerinin kütle/boyut parametreleri anlık çekilir.</p></div>
        <div class="sg-card"><div class="sg-step-n">02</div><div class="sg-ct">Önbellek ve Veri İşleme</div><p class="sg-cp"><b style="color:#cbd5e1">Data &amp; Cache</b> — API limitlerini aşmamak için veriler JSON/Redis önbelleğinde saklanır, veri birleştirici modülümüz ile temizlenerek hesaplama motoruna beslenir.</p></div>
        <div class="sg-card"><div class="sg-step-n">03</div><div class="sg-ct">Astrodinamik ve Finansal Risk Motoru</div><p class="sg-cp"><b style="color:#cbd5e1">Calculations</b> — SGP4 motoru ile koordinatlar hesaplanır, mesafe tarayıcı 50 km yakınlaşmaları tespit eder ve Finansal Risk Motoru anlık risk skorunu üretir.</p></div>
        <div class="sg-card"><div class="sg-step-n">04</div><div class="sg-ct">Sunum ve Alarm Paneli</div><p class="sg-cp"><b style="color:#cbd5e1">Presentation</b> — 3D Canlı Dünya Görselleştirici, Çarpışma Alarm Paneli ve B2B SaaS Fiyat Tabloları ile veriler karar vericilere sunulur.</p></div>
      </div>
      <div class="sg-src" style="margin-top:22px">API kaynakları <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
    </div>

    <div id="pricing" class="sg-sec">
      <div class="sg-kicker">B2B Fiyatlandırma</div>
      <div class="sg-h2">İhtiyacınıza uygun B2B SaaS paketleri</div>
      <p class="sg-p">Farklı paydaş gruplarının ihtiyaçlarına özel kurgulanmış <b>3 kademeli (Tiered)</b> B2B fiyatlandırma modeli:</p>
      <div class="sg-price">
        <div class="sg-card sg-tier" style="--c:#3b82f6">
          <div class="sg-tier-name">BASIC/STANDART</div>
          <div><div class="sg-lbl">Hedef Kitle</div><div class="sg-val"><span style="color:var(--c)">☆</span> Erken aşama ve küçük ölçekli uydu takımyıldızı operatörleri.<br><span style="color:var(--c)">☆</span> Uzay teknolojileri Ar-Ge merkezleri ve akademik araştırma kurumları.</div></div>
          <div><div class="sg-lbl">Sigorta ve Risk Kapsamı</div><div class="sg-val">Standart Yörünge Risk Değerlendirme Karnesi <span style="color:var(--c)">✓</span><br>Sigorta süreçleri için temel çarpışma olasılığı ve enkaz analiz raporu <span style="color:var(--c)">✓</span></div></div>
        </div>
        <div class="sg-card sg-tier pop" style="--c:#06b6d4">
          <span class="sg-tag">Popüler</span>
          <div class="sg-tier-name">PRO / ENTERPRISE</div>
          <div><div class="sg-lbl">Hedef Kitle</div><div class="sg-val"><span style="color:var(--c)">☆</span> Kesintisiz operasyon hedefleyen ticari uydu operatörleri<br><span style="color:var(--c)">☆</span> Yüksek yörünge trafiği yöneten filo operasyon merkezleri</div></div>
          <div><div class="sg-lbl">Sigorta ve Risk Kapsamı</div><div class="sg-val">Poliçe prim optimizasyonu sağlayan manevra ve yakıt verimliliği dokümantasyonu <span style="color:var(--c)">✓</span><br>Olası çarpışma ve hizmet kesintisi (BI) finansal etki projeksiyonu <span style="color:var(--c)">✓</span></div></div>
        </div>
        <div class="sg-card sg-tier" style="--c:#a855f7">
          <div class="sg-tier-name">INSTITUTIONAL / INSURTECH</div>
          <div><div class="sg-lbl">Hedef Kitle</div><div class="sg-val"><span style="color:var(--c)">☆</span> Ulusal uzay ajansları, kamu otoriteleri ve savunma sanayi kuruluşları<br><span style="color:var(--c)">☆</span> Uzay varlıklarını teminat altına alan sigorta ve reasürans şirketleri</div></div>
          <div><div class="sg-lbl">Sigorta ve Risk Kapsamı</div><div class="sg-val">Dinamik poliçe fiyatlandırma ve reasürans risk skorlaması <span style="color:var(--c)">✓</span><br>Kaza sonrası adli yörünge doğrulama ve kümülatif portföy simülasyonu <span style="color:var(--c)">✓</span></div></div>
        </div>
      </div>
      <div class="sg-faq">
        <div class="sg-kicker" style="margin-bottom:16px">Sık sorulanlar</div>
        <details><summary>1. Space Grave yörünge verilerini ne kadar sıklıkla güncelliyor?</summary><p>Platformumuz CelesTrak, Space-Track ve ESA DISCOS API'leri üzerinden canlı TLE (Two-Line Element) verilerini düzenli olarak çeker. API sorgu limitlerini korumak ve maksimum performans sağlamak amacıyla veriler JSON/Redis önbelleğimizde saklanır ve otomatik senkronize edilir.</p></details>
        <details><summary>2. 50 km Güvenlik Küresi (Safety Sphere) ne anlama geliyor?</summary><p>Uydunuzun etrafında sanal olarak tanımlanan 50 km yarıçapındaki koruma hacmidir. SGP4 astrodinamik motorumuz bir uzay çöpünün bu küreye yaklaşacağını öngördüğü anda sistem "Kırmızı Alarm" durumuna geçer ve anlık kaçınma analizi başlatır.</p></details>
        <details><summary>3. Dinamik Sigorta Risk Skoru (0-100) nasıl hesaplanıyor?</summary><p>Risk skoru; uydunun bağıl hızı, çarpışma olasılığı ve kinetik enerjisi kullanılarak kurgulanır. NASA'nın 40 J/g parçalanma eşiği baz alınarak; roket gövdeleri veya yüksek yoğunluklu enkaz parçaları için kütle çarpanları uygulanır ve 0-100 arası finansal poliçe skoruna dönüştürülür.</p></details>
        <details><summary>4. Kendi özel uydu filomuzun verilerini sisteme entegre edebilir miyiz?</summary><p>Evet. Basic/Standart paketimizde kataloglanmış genel yörünge takibi sunulurken; Pro/Enterprise ve Institutional/InsurTech paketlerimizde kendi aktif uydu filolarınızın özel Ephemeris ve TLE verilerini güvenli API çağrılarıyla sisteme aktarabilirsiniz.</p></details>
        <details><summary>5. Platform yanlış alarmları (False Positive) ve gereksiz yakıt harcamasını nasıl önlüyor?</summary><p>Önbellek ve veri temizleme katmanımız (Data Pipeline), gürültülü verileri ve sapmaları filtreler. Yalnızca kütle, hız ve mesafe açısından NASA parçalanma standartlarını aşan gerçek tehlikeleri alarm olarak bildirdiği için operatörlerin gereksiz manevra yakıtı harcamasını engeller.</p></details>
      </div>
    </div>

    <div id="about" class="sg-sec">
      <div class="sg-about">
        <div>
          <div class="sg-kicker">Hakkımızda</div>
          <div class="sg-h2">Karmaşık yörünge verisini karar verilebilir riske dönüştürüyoruz</div>
          <p class="sg-p">Space Grave, NASA Space Apps Challenge kapsamında geliştirilen bir B2B karar destek platformudur. Amacımız; uydu operatörlerinin ve uzay sigortacılarının uzay çöpü kaynaklı riski gerçekleşmeden görmesini, gereksiz manevra maliyetinden kaçınmasını ve zararlarını minimuma indirmesini sağlamaktır.</p>
        </div>
        <div class="sg-card"><div class="sg-stat sg-huge">&#36;42.3B</div><div class="sg-ct">Gizli finansal yük</div><p class="sg-cp">Uzay çöpünün LEO operatörlerine yüklediği tahmini ekonomik etki — WEF risk verisi.</p></div>
      </div>
    </div>

    <div class="sg-sec sg-final">
      <div class="sg-h2">Canlı yörünge haritasına geçin</div>
      <p class="sg-p">Gerçek TLE verisiyle 3D Dünya, yakınlaşma taraması ve atmosfere yeniden giriş simülasyonu tek komuta merkezinde.</p>
    </div>
    """

    foot_html = f"""
    <div class="sg-foot">NASA Space Apps Challenge · JPL Command Center · SP4C3 GR4V3 &nbsp;|&nbsp; Veri: CelesTrak · Son güncelleme {CAT['fetched']}</div>
    """


    if st.session_state.lang == "en":
        nav_html = """
        <div class="sg-nav">
          <a class="sg-logo" href="#top"><span class="sg-logo-mark"></span>SPACE GRAVE</a>
          <div class="sg-links"><a href="#features">Features</a><a href="#pricing">B2B Pricing</a><a href="#docs">Specs / Documentation</a><a href="#about">About Us</a></div>
          <a class="sg-cta" href="?launch=1&amp;lang=en" target="_self">Start Live Tracking</a>
        </div>
        <div class="sg-veil"></div>
        <a class="sg-top" href="#top" aria-label="Back to top">↑ Top</a>
        """
        hero_html = f"""
        <div id="top" class="sg-hero">
          <div class="sg-badge"><span class="sg-dot{'' if feed_word == 'LIVE' else ' warn'}"></span>B2B Space Decision Support Platform</div>
          <div class="sg-h1">Space Grave: <em>smart collision avoidance</em> and a decision support platform for Low Earth Orbit (LEO)</div>
          <p class="sg-lead">With the Space Grave decision support system, protect your satellite fleets inside a 50 km safety sphere from space debris and foresee catastrophic risk before it happens.</p>
        </div>
        """
        metrics_html = f"""
        <div class="sg-metrics">
          <div class="sg-metric"><b>{_fmt_int(n_total)}</b><span>Tracked objects</span></div>
          <div class="sg-metric"><b>50 km</b><span>Safety sphere</span></div>
          <div class="sg-metric"><b>40 J/g</b><span>NASA fragmentation threshold</span></div>
          <div class="sg-metric"><b class="{'ok' if feed_word == 'LIVE' else 'warn'}">{feed_word}</b><span>CelesTrak data feed</span></div>
        </div>
        <div class="sg-src">Trusted data sources <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
        """
        sections_html = """
        <div id="threat" class="sg-sec sg-first">
          <div class="sg-kicker">Space Debris</div>
          <div class="sg-h2">The Silent Threat in Orbit: Space Debris</div>
          <p class="sg-p">Since the first satellite launches, spacecraft at the end of their missions and spent upper rocket stages have been left in orbit; the number of cataloged objects reached 5,600 by 1990. On 10 February 2009 a turning point came: Iridium 33 and Cosmos 2251 collided at an altitude of 790 kilometers. The event went on record as the first accidental hypervelocity collision between two satellites. In that single event, about 200,000 small untrackable fragments and more than 1,800 large pieces were scattered into orbit. More than 6,000 defunct satellites and rocket bodies drifting in orbit raise the risk of a chain-reaction collision every year for active commercial fleets and insurance portfolios.</p>
        </div>

        <div id="features" class="sg-sec">
          <div class="sg-kicker">Features</div>
          <div class="sg-h2">Space debris is no longer just a technical risk, it is a financial one</div>
          <p class="sg-p">Space debris is not only a technical threat for satellite fleets operating in Low Earth Orbit (LEO); it is also a "hidden financial burden" worth <b>&#36;42.3B</b>. Space Grave processes complex orbital data to give satellite operators and space insurers instant risk foresight and cost optimization.</p>
          <div class="sg-grid">
            <div class="sg-card"><div class="sg-stat">50 km</div><div class="sg-ct">Safety Sphere</div><p class="sg-cp">Automatically triggers a Red Alert on high-risk close approaches.</p></div>
            <div class="sg-card"><div class="sg-stat">SGP4</div><div class="sg-ct">Astrodynamics Engine and 3D Visualizer</div><p class="sg-cp">Coordinates computed from live TLE data are tracked on a live 3D Earth visualizer; the distance scanner detects close approaches.</p></div>
            <div class="sg-card"><div class="sg-stat">40 J/g</div><div class="sg-ct">NASA Fragmentation Standard</div><p class="sg-cp">Predicts destruction potential with kinetic energy calculations.</p></div>
            <div class="sg-card"><div class="sg-stat">0–100</div><div class="sg-ct">Insurance Risk Score</div><p class="sg-cp">Provides dynamic policy analysis for rocket bodies and high-density debris.</p></div>
          </div>
          <p class="sg-p sg-muted" style="margin-top:28px">Against the &#36;42.3B space debris threat, we process live TLE data with the SGP4 engine and NASA fragmentation standards to scan for debris inside the 50 km safety sphere; by instantly calculating a dynamic 0-100 insurance risk score and avoidance costs, we help you optimize maneuver fuel costs and minimize service interruptions and losses.</p>
        </div>

        <div id="docs" class="sg-sec">
          <div class="sg-kicker">Specs / Documentation</div>
          <div class="sg-h2">Modular and scalable system architecture</div>
          <p class="sg-p">Our platform has a high-performance infrastructure made up of 4 main layers, from pulling live data to presenting it in the 3D interface:</p>
          <div class="sg-grid">
            <div class="sg-card"><div class="sg-step-n">01</div><div class="sg-ct">Data Collection Layer</div><p class="sg-cp"><b style="color:#cbd5e1">Data Ingestion</b> — Live TLE orbit data and debris mass/size parameters are pulled instantly from the CelesTrak, Space-Track and ESA DISCOS APIs.</p></div>
            <div class="sg-card"><div class="sg-step-n">02</div><div class="sg-ct">Cache and Data Processing</div><p class="sg-cp"><b style="color:#cbd5e1">Data &amp; Cache</b> — To stay within API limits, data is stored in a JSON/Redis cache, cleaned by our data merger module and fed to the calculation engine.</p></div>
            <div class="sg-card"><div class="sg-step-n">03</div><div class="sg-ct">Astrodynamics and Financial Risk Engine</div><p class="sg-cp"><b style="color:#cbd5e1">Calculations</b> — The SGP4 engine computes coordinates, the distance scanner detects 50 km close approaches and the Financial Risk Engine produces an instant risk score.</p></div>
            <div class="sg-card"><div class="sg-step-n">04</div><div class="sg-ct">Presentation and Alert Panel</div><p class="sg-cp"><b style="color:#cbd5e1">Presentation</b> — Data reaches decision makers through the 3D Live Earth Visualizer, the Collision Alert Panel and B2B SaaS pricing tables.</p></div>
          </div>
          <div class="sg-src" style="margin-top:22px">API sources <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
        </div>

        <div id="pricing" class="sg-sec">
          <div class="sg-kicker">B2B Pricing</div>
          <div class="sg-h2">B2B SaaS plans that fit your needs</div>
          <p class="sg-p">A <b>3-tier</b> B2B pricing model designed for the needs of different stakeholder groups:</p>
          <div class="sg-price">
            <div class="sg-card sg-tier" style="--c:#3b82f6">
              <div class="sg-tier-name">Basic / Standard</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Small constellation operators<br>Universities / research</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Maneuver cost and fuel optimizer<br>Cataloged debris tracking</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Flat-rate: monthly or annual subscription per satellite / constellation</div></div>
            </div>
            <div class="sg-card sg-tier pop" style="--c:#06b6d4">
              <span class="sg-tag">Popular</span>
              <div class="sg-tier-name">Pro / Enterprise</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Commercial satellite operators<br>Fleet managers</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Anomaly detection and risk forecasting<br>Outage duration / cost analysis</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Number of active satellites monitored and real-time API calls</div></div>
            </div>
            <div class="sg-card sg-tier" style="--c:#a855f7">
              <div class="sg-tier-name">Institutional / InsurTech</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Defense and space agencies<br>Insurance companies</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Cumulative financial risk and scenario model<br>Custom risk scoring and policy pricing</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Custom annual contract + setup + consulting fee</div></div>
            </div>
          </div>
          <div class="sg-faq">
            <div class="sg-kicker" style="margin-bottom:16px">FAQ</div>
            <details><summary>1. How often does Space Grave update orbit data?</summary><p>Our platform regularly pulls live TLE (Two-Line Element) data through the CelesTrak, Space-Track and ESA DISCOS APIs. To respect API query limits and ensure maximum performance, data is stored in our JSON/Redis cache and synced automatically.</p></details>
            <details><summary>2. What does the 50 km Safety Sphere mean?</summary><p>It is a protection volume with a 50 km radius virtually defined around your satellite. The moment our SGP4 astrodynamics engine predicts that a piece of debris will approach this sphere, the system switches to "Red Alert" and starts an instant avoidance analysis.</p></details>
            <details><summary>3. How is the Dynamic Insurance Risk Score (0-100) calculated?</summary><p>The risk score is built from the satellite's relative velocity, collision probability and kinetic energy. Based on NASA's 40 J/g fragmentation threshold, mass multipliers are applied for rocket bodies and high-density debris, and the result is converted into a 0-100 financial policy score.</p></details>
            <details><summary>4. Can we integrate our own private satellite fleet data into the system?</summary><p>Yes. The Basic/Standard plan offers cataloged general orbit tracking; with the Pro/Enterprise and Institutional/InsurTech plans you can send your own active fleets' private Ephemeris and TLE data to the system through secure API calls.</p></details>
            <details><summary>5. How does the platform prevent false positives and unnecessary fuel use?</summary><p>Our cache and data cleaning layer (Data Pipeline) filters noisy data and deviations. Because only real hazards that exceed NASA fragmentation standards in mass, velocity and distance are raised as alerts, operators avoid spending maneuver fuel unnecessarily.</p></details>
          </div>
        </div>

        <div id="about" class="sg-sec">
          <div class="sg-about">
            <div>
              <div class="sg-kicker">About Us</div>
              <div class="sg-h2">We turn complex orbital data into actionable risk</div>
              <p class="sg-p">Space Grave is a B2B decision support platform developed for the NASA Space Apps Challenge. Our goal is to let satellite operators and space insurers see debris-related risk before it happens, avoid unnecessary maneuver costs and minimize their losses.</p>
            </div>
            <div class="sg-card"><div class="sg-stat sg-huge">&#36;42.3B</div><div class="sg-ct">Hidden financial burden</div><p class="sg-cp">Estimated economic impact of space debris on LEO operators — WEF risk data.</p></div>
          </div>
        </div>

        <div class="sg-sec sg-final">
          <div class="sg-h2">Move on to the live orbit map</div>
          <p class="sg-p">3D Earth with real TLE data, close-approach screening and atmospheric re-entry simulation in a single command center.</p>
        </div>
        """
        foot_html = f"""
        <div class="sg-foot">NASA Space Apps Challenge · JPL Command Center · SP4C3 GR4V3 &nbsp;|&nbsp; Data: CelesTrak · Last update {CAT['fetched']}</div>
        """

    _cut = sections_html.index('<div id="about"')
    sections_html, tail_html = sections_html[:_cut], sections_html[_cut:]

    # ISS canlı yayını: tıklanınca yüklenen hafif oynatıcı (açılışta hiçbir YouTube kaynağı indirilmez)
    if st.session_state.lang == "en":
        live_html = """
        <div id="live" class="sg-sec">
          <div class="sg-kicker">Live from orbit</div>
          <div class="sg-h2">Watch the orbit we protect: live ISS view</div>
          <p class="sg-p">The International Space Station circles Earth about 420 km up, in the same Low Earth Orbit region where Space Grave screens debris and close approaches, completing one orbit roughly every 90 minutes. Press play to watch the NASA external HD camera feed live and see the environment our platform is built to keep safe.</p>
        </div>
        """
        live_play, live_cap = "Play live stream", "Live ISS view · NASA external cameras via afarTV"
    else:
        live_html = """
        <div id="live" class="sg-sec">
          <div class="sg-kicker">Yörüngeden canlı</div>
          <div class="sg-h2">Koruduğumuz yörüngeyi canlı izleyin: ISS yayını</div>
          <p class="sg-p">Uluslararası Uzay İstasyonu, Space Grave'in uzay çöpü ve yakın geçişleri taradığı Alçak Dünya Yörüngesi'nde (LEO), yaklaşık 420 km yükseklikte ve yaklaşık her 90 dakikada bir tur atar. Oynat'a basarak NASA'nın dış HD kamera yayınını canlı izleyin ve platformumuzun güvende tutmak için tasarlandığı ortamı yakından görün.</p>
        </div>
        """
        live_play, live_cap = "Canlı yayını oynat", "Canlı ISS görüntüsü · NASA dış kameraları (afarTV)"

    st.markdown(INTRO_CSS, unsafe_allow_html=True)
    with (st.container(key="sg_fg") if keyed else st.container()):
        st.markdown(_flat(nav_html), unsafe_allow_html=True)
        left, _ = st.columns([1.25, 1])
        with left:
            st.markdown(_flat(hero_html), unsafe_allow_html=True)
            b1, b2 = st.columns([1.15, 1])
            with b1:
                st.button(T("Canlı İzlemeyi Başlat  →", "Start Live Tracking  →"), key="sg_launch", type="primary", on_click=_enter_app, **stretch(st.button))
            with b2:
                st.markdown(f'<a class="sg-ghost" href="#features">{T("Özellikleri keşfet", "Explore features")}</a>', unsafe_allow_html=True)
            st.markdown(_flat(metrics_html), unsafe_allow_html=True)
        st.markdown(_flat(sections_html), unsafe_allow_html=True)
        st.markdown(_flat(_debris_history_html()), unsafe_allow_html=True)
        st.markdown(_flat(live_html), unsafe_allow_html=True)
        _live_player_ui(live_play, live_cap)
        st.markdown(_flat(tail_html), unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.1, 1])
        with c2:
            st.button(T("Canlı İzlemeyi Başlat  →", "Start Live Tracking  →"), key="sg_launch_2", type="primary", on_click=_enter_app, **stretch(st.button))
        st.markdown(_flat(foot_html), unsafe_allow_html=True)
    render_lang_switch("24px", "12px")


if not st.session_state.intro_passed and _launch_from_url():
    st.session_state.intro_passed = True
    st.session_state.sg_just_entered = True
    try:
        st.query_params.clear()
    except Exception:
        pass

if not st.session_state.intro_passed:
    render_intro()
    st.stop()

# Girişten ana sayfaya geçerken eski (bayat) giriş öğeleri soluk yazı olarak kalmasın
if st.session_state.pop('sg_just_entered', False):
    st.markdown("<style>[data-stale='true']{display:none !important}</style>", unsafe_allow_html=True)

# --- ANA KOMUTA MERKEZİ ---
MAIN_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp {
  font-family:'Inter','Segoe UI',-apple-system,sans-serif;
  background:
    radial-gradient(1.2px 1.2px at 8% 14%, rgba(255,255,255,.55), transparent 100%),
    radial-gradient(1px 1px at 23% 61%, rgba(186,230,253,.5), transparent 100%),
    radial-gradient(1.4px 1.4px at 37% 27%, rgba(255,255,255,.45), transparent 100%),
    radial-gradient(1px 1px at 52% 83%, rgba(255,255,255,.5), transparent 100%),
    radial-gradient(1.2px 1.2px at 66% 9%, rgba(186,230,253,.55), transparent 100%),
    radial-gradient(1px 1px at 79% 47%, rgba(255,255,255,.45), transparent 100%),
    radial-gradient(1.3px 1.3px at 91% 72%, rgba(255,255,255,.5), transparent 100%),
    radial-gradient(1px 1px at 14% 92%, rgba(186,230,253,.45), transparent 100%),
    radial-gradient(ellipse at 78% -8%, #0f2347 0%, #060d22 42%, #02040d 100%) !important;
  background-attachment: fixed !important;
}
.block-container, [data-testid="stMainBlockContainer"] { padding-top:5.6rem !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family:'Space Grotesk','Inter','Segoe UI',sans-serif !important; letter-spacing:-.015em; }
.stApp h1 { color:#fff; font-weight:700 !important; }
.stApp h1, .stApp h3 { text-shadow:0 0 28px rgba(56,189,248,.18); }

/* Üst gezinme çubuğu (giriş ekranıyla aynı dil) */
.sg-mnav { position:fixed; top:0; left:0; right:0; height:64px; z-index:50; display:flex; align-items:center; justify-content:space-between;
  padding:0 clamp(16px,3vw,40px) 0 calc(clamp(16px,3vw,40px) + 168px); background:linear-gradient(180deg,rgba(4,8,23,.92),rgba(4,8,23,.62));
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border-bottom:1px solid rgba(56,189,248,.16); }
.sg-mlogo { display:flex; align-items:center; gap:11px; color:#fff; font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:.16em; font-size:15px; }
.sg-logo-mark { width:22px; height:22px; border-radius:50%; border:1.5px solid #38bdf8; position:relative; box-shadow:0 0 12px rgba(56,189,248,.4); display:inline-block; }
.sg-logo-mark::after { content:""; position:absolute; width:5px; height:5px; border-radius:50%; background:#fc3d21; top:-3px; right:1px; }
.sg-mlive { display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; border:1px solid rgba(56,189,248,.16);
  background:rgba(8,16,38,.65); color:#bae6fd; font-size:11.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
.sg-dot { width:7px; height:7px; border-radius:50%; background:#10b981; animation:sgPulse 2.2s infinite; }
.sg-dot.warn { background:#f59e0b; animation:none; }
@keyframes sgPulse { 0% { box-shadow:0 0 0 0 rgba(16,185,129,.55); } 70% { box-shadow:0 0 0 9px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@media (max-width:700px) { .sg-mlogo { display:none; } .sg-mnav { padding-left:calc(clamp(16px,3vw,40px) + 168px); } }

/* Sol üstteki geri butonu */
.st-key-sg_back { position:fixed; top:11px; left:clamp(16px,3vw,40px); z-index:60; width:auto !important; }
.st-key-sg_back .stButton > button { height:42px; padding:0 20px; border-radius:999px; white-space:nowrap;
  border:1px solid rgba(56,189,248,.55); background:rgba(56,189,248,.10); box-shadow:none; transition:all .2s ease; }
.st-key-sg_back .stButton > button p { color:#fff; font-size:13px; font-weight:600; margin:0; }
.st-key-sg_back .stButton > button:hover { background:rgba(56,189,248,.24); border-color:#38bdf8; transform:translateY(-1px); }

/* Modül butonları */
[class*="st-key-sg_nav_"] .stButton > button { border-radius:999px; border:1px solid rgba(56,189,248,.28); background:rgba(8,16,38,.70);
  min-height:46px; transition:all .2s ease; backdrop-filter:blur(10px); }
[class*="st-key-sg_nav_"] .stButton > button p { color:#cbd5e1; font-weight:600; font-size:14px; }
[class*="st-key-sg_nav_"] .stButton > button:hover { border-color:#38bdf8; background:rgba(56,189,248,.14); transform:translateY(-2px); }

/* Kartlar, tablolar, grafikler */
.nasa-glass-card:hover { transform:translateY(-3px); border-color:rgba(56,189,248,.5); }
[data-testid="stMetric"] { background:linear-gradient(160deg,rgba(14,24,48,.80),rgba(6,11,26,.90)); border:1px solid rgba(56,189,248,.16);
  border-radius:16px; padding:14px 18px; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border:1px solid rgba(56,189,248,.14); border-radius:14px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:14px; }
.stApp hr { border-color:rgba(56,189,248,.18) !important; }
.stApp [data-baseweb="select"] > div { border-radius:12px; }
.stApp .stButton > button[kind="primary"] { border-radius:999px; transition:transform .2s ease, box-shadow .2s ease; }
.stApp .stButton > button[kind="primary"]:hover { transform:translateY(-2px); }
.stApp iframe { border-radius:16px; }
@media (prefers-reduced-motion:reduce) { .sg-dot { animation:none; } }
</style>
"""


def _back_to_intro():
    st.session_state.intro_passed = False


st.markdown(MAIN_CSS, unsafe_allow_html=True)

# --- OPS TEMASI: NASA/JPL/ESA komuta merkezi estetiği (MAIN_CSS'in üzerine biner) ---
OPS_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root{--ops-bg:#050B14;--ops-cy:#22d3ee;--ops-cy2:#67e8f9;--ops-amb:#f59e0b;--ops-red:#ef4444;--ops-grn:#34d399;--ops-mute:#8195b0;--ops-line:rgba(34,211,238,.16);--ops-card:rgba(8,16,30,.72);--ops-mono:'JetBrains Mono',ui-monospace,Consolas,monospace}

/* Derin uzay zemini + ince koordinat ızgarası */
.stApp{
  background:
    linear-gradient(rgba(34,211,238,.028) 1px,transparent 1px) 0 0/48px 48px,
    linear-gradient(90deg,rgba(34,211,238,.028) 1px,transparent 1px) 0 0/48px 48px,
    radial-gradient(ellipse at 78% -10%,#0a1a33 0%,#050B14 46%,#03070e 100%) !important;
  background-attachment:fixed !important;
}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:7.6rem !important}

/* Üst gezinme çubuğu */
.sg-mnav{background:linear-gradient(180deg,rgba(5,11,20,.97),rgba(5,11,20,.86)) !important;border-bottom:1px solid var(--ops-line) !important}
.sg-mlive{border-radius:4px !important;font-family:var(--ops-mono);font-size:10.5px !important;border-color:var(--ops-line) !important;background:rgba(8,16,30,.8) !important}

/* Kayan telemetri şeridi (iframe, sabit) */
.st-key-sg_ticker{position:fixed !important;top:64px;left:0;right:0;z-index:49;width:100vw !important;margin:0 !important;padding:0 !important}
.st-key-sg_ticker>div,.st-key-sg_ticker .element-container,.st-key-sg_ticker [data-testid="stElementContainer"],.st-key-sg_ticker [data-testid="stIFrame"],.st-key-sg_ticker iframe{width:100vw !important;height:34px !important;max-width:none !important;margin:0 !important;border:0 !important;border-radius:0 !important}

/* Görev başlığı */
.ops-banner{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;padding:4px 0 14px;margin-bottom:4px;border-bottom:1px solid var(--ops-line);position:relative}
.ops-banner::after{content:"";position:absolute;left:0;bottom:-1px;width:140px;height:2px;background:linear-gradient(90deg,var(--ops-cy),transparent)}
.ops-eyebrow{font-family:var(--ops-mono);font-size:11px;letter-spacing:.22em;color:var(--ops-cy);text-transform:uppercase}
.ops-title{font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:clamp(24px,2.8vw,38px);letter-spacing:.02em;color:#f1f5f9;margin:6px 0 2px;line-height:1.1}
.ops-title span{color:var(--ops-cy);text-shadow:0 0 22px rgba(34,211,238,.45)}
.ops-sub{font-size:13px;color:var(--ops-mute);letter-spacing:.04em}
.ops-leds{display:flex;gap:8px;flex-wrap:wrap}
.ops-led{display:inline-flex;align-items:center;gap:8px;padding:6px 11px;border:1px solid var(--ops-line);border-radius:3px;background:var(--ops-card);font-family:var(--ops-mono);font-size:10.5px;letter-spacing:.14em;color:#cbd5e1;text-transform:uppercase}
.ops-led::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--c,#34d399);box-shadow:0 0 8px var(--c,#34d399);animation:opsPulse 2s infinite}
.ops-led.g{--c:#34d399}.ops-led.c{--c:#22d3ee}.ops-led.a{--c:#f59e0b}.ops-led.r{--c:#ef4444}
@keyframes opsPulse{0%,100%{opacity:1}50%{opacity:.35}}

/* Durum şeridi */
.ops-strip{display:flex;flex-wrap:wrap;gap:0;margin:10px 0 6px;border:1px solid var(--ops-line);border-radius:4px;background:var(--ops-card);overflow:hidden;font-family:var(--ops-mono)}
.ops-strip span{display:flex;flex-direction:column;gap:2px;padding:8px 16px;border-right:1px solid var(--ops-line)}
.ops-strip span:last-child{border-right:0}
.ops-strip i{font-style:normal;font-size:9.5px;letter-spacing:.2em;color:var(--ops-mute);text-transform:uppercase}
.ops-strip b{font-size:12.5px;font-weight:500;color:var(--ops-cy2)}

/* Modül sekmeleri */
[class*="st-key-sg_nav_"] .stButton>button{border-radius:3px !important;border:1px solid var(--ops-line) !important;background:rgba(8,16,30,.78) !important;min-height:42px !important;transform:none !important}
[class*="st-key-sg_nav_"] .stButton>button p{font-family:var(--ops-mono) !important;font-size:12px !important;letter-spacing:.06em;color:#9fb3cc !important;font-weight:500 !important}
[class*="st-key-sg_nav_"] .stButton>button:hover{border-color:var(--ops-cy) !important;background:rgba(34,211,238,.08) !important}

/* İstatistik kartları: kompakt, sol hizalı, köşe işaretli */
.stApp .nasa-glass-card{text-align:left !important;padding:12px 14px 12px 16px !important;border-radius:4px !important;margin-bottom:8px !important;
  background:linear-gradient(160deg,rgba(10,20,36,.88),rgba(5,11,20,.94)) !important;border:1px solid var(--ops-line) !important;
  box-shadow:inset 0 0 24px rgba(34,211,238,.04),0 6px 20px rgba(0,0,0,.45) !important;backdrop-filter:blur(10px)}
.stApp .nasa-glass-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--ops-cy);box-shadow:0 0 10px var(--ops-cy)}
.stApp .nasa-glass-card.fin-card{border-color:rgba(245,158,11,.3) !important}
.stApp .nasa-glass-card.fin-card::before{background:var(--ops-amb);box-shadow:0 0 10px var(--ops-amb)}
.stApp .nasa-glass-card:hover{transform:none !important;border-color:rgba(34,211,238,.5) !important}
.stApp .metric-title{font-family:var(--ops-mono) !important;font-size:9.5px !important;letter-spacing:.2em !important;color:var(--ops-mute) !important;font-weight:500 !important}
.stApp .metric-value,.stApp .metric-value-fin{font-family:var(--ops-mono) !important;font-size:22px !important;font-weight:700 !important;letter-spacing:-.01em;margin-top:4px !important;font-variant-numeric:tabular-nums}
.stApp .metric-value{text-shadow:0 0 14px rgba(34,211,238,.45) !important}
.stApp .metric-value-fin{text-shadow:0 0 14px rgba(245,158,11,.4) !important}
[data-testid="stMetric"]{border-radius:4px !important;background:var(--ops-card) !important;border-color:var(--ops-line) !important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{font-family:var(--ops-mono) !important}

/* Bölüm başlıkları */
.stApp h4{font-family:'Space Grotesk','Inter',sans-serif !important;font-size:14px !important;font-weight:600 !important;letter-spacing:.14em !important;text-transform:uppercase;color:#e2e8f0;
  padding:2px 0 2px 12px;border-left:2px solid var(--ops-cy);margin:20px 0 10px !important}
.stApp h3{font-family:'Space Grotesk','Inter',sans-serif !important}

/* Grafik / tablo çerçeveleri */
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{border:1px solid var(--ops-line) !important;border-radius:4px !important;background:var(--ops-card);backdrop-filter:blur(8px)}
.stApp iframe{border-radius:4px}
[data-testid="stAlert"]{border-radius:4px !important}
.stApp hr{border-color:var(--ops-line) !important}

/* Operasyon tablosu */
.ops-tbl-bar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 12px;border:1px solid var(--ops-line);border-bottom:0;border-radius:4px 4px 0 0;background:rgba(8,16,30,.9);font-family:var(--ops-mono);font-size:10.5px;letter-spacing:.14em;color:var(--ops-mute);text-transform:uppercase}
.ops-tbl-bar b{color:var(--ops-cy2);font-weight:500}
.ops-tbl-wrap{max-height:400px;overflow:auto;border:1px solid var(--ops-line);border-radius:0 0 4px 4px;background:rgba(5,11,20,.82);scrollbar-width:thin;scrollbar-color:#17314f #050B14}
.ops-tbl{width:100%;border-collapse:collapse;font-family:var(--ops-mono);font-size:11.5px;color:#cbd5e1}
.ops-tbl th{position:sticky;top:0;z-index:2;background:#08111f;color:var(--ops-cy2);font-weight:500;font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(34,211,238,.3);white-space:nowrap}
.ops-tbl td{padding:5px 10px;border-bottom:1px solid rgba(34,211,238,.07);white-space:nowrap}
.ops-tbl tbody tr:nth-child(even){background:rgba(34,211,238,.022)}
.ops-tbl tbody tr:hover{background:rgba(34,211,238,.09)}
.ops-tbl .n{text-align:right;font-variant-numeric:tabular-nums}
.ops-tbl th.n{text-align:right}
.ops-tbl td.nm{font-family:'Inter',sans-serif;font-weight:500;color:#f1f5f9}
.ops-chip{display:inline-block;padding:2px 8px;border-radius:2px;font-size:10px;letter-spacing:.08em;border:1px solid;font-weight:500}
.ops-chip.ok{color:#34d399;border-color:rgba(52,211,153,.5);background:rgba(52,211,153,.08)}
.ops-chip.warn{color:#f59e0b;border-color:rgba(245,158,11,.5);background:rgba(245,158,11,.08)}
.ops-chip.nom{color:#22d3ee;border-color:rgba(34,211,238,.4);background:rgba(34,211,238,.06)}
.ops-chip.crit{color:#ef4444;border-color:rgba(239,68,68,.5);background:rgba(239,68,68,.08)}
@media (max-width:700px){.ops-strip span{padding:6px 10px}}
</style>
"""
st.markdown(OPS_CSS, unsafe_allow_html=True)

# Tüm plotly grafiklerine (template="plotly_dark") ortak operasyon stili
try:
    import plotly.io as pio
    pio.templates["plotly_dark"].layout.update(
        font=dict(family="JetBrains Mono, Consolas, monospace", color="#8195b0", size=11),
        colorway=["#22d3ee", "#f59e0b", "#ef4444", "#34d399", "#a78bfa", "#67e8f9"],
        xaxis=dict(gridcolor="rgba(34,211,238,.10)", linecolor="rgba(34,211,238,.35)", zerolinecolor="rgba(34,211,238,.18)"),
        yaxis=dict(gridcolor="rgba(34,211,238,.10)", linecolor="rgba(34,211,238,.35)", zerolinecolor="rgba(34,211,238,.18)"),
        hoverlabel=dict(bgcolor="#0a1424", bordercolor="#22d3ee", font=dict(family="JetBrains Mono, Consolas, monospace", color="#e2e8f0")),
    )
except Exception:
    pass
# =====================================================================
# SP4C3 GR4V3 — YENİDEN GİRİŞ SİMÜLASYON MODU (izole / odak modu)
# Simülasyon açıkken aşağıdaki sitenin geri kalanı (navigasyon, sayfalar, bot, taramalar)
# HİÇ ÇALIŞTIRILMAZ: Python st.stop() ile durur, fizik ve çizim tarayıcıda (iframe/JS) koşar.
# Böylece Streamlit yeniden çalıştırması olmaz, diğer modüllerin durumu (session_state) aynen korunur.
# =====================================================================
ATM = [(0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554),
       (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799),
       (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
       (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523)]
MATERIALS = {"Aluminum alloy": 0.97, "Stainless steel": 1.0, "Titanium": 1.15, "Carbon composite (CFRP/C-C)": 3.0}
_MAT_TR = {"Aluminum alloy": "Alüminyum alaşımı", "Stainless steel": "Paslanmaz çelik", "Titanium": "Titanyum",
           "Carbon composite (CFRP/C-C)": "Karbon kompozit (CFRP/C-C)"}
REGIONS = [["Point Nemo (SPOUA) · kontrollü okyanus", "Point Nemo (SPOUA) · controlled ocean", -48.9, -123.4, 0.1],
           ["Kuzey Pasifik açık deniz", "North Pacific open ocean", 30.0, -150.0, 0.1],
           ["Güney Atlantik", "South Atlantic", -35.0, -20.0, 0.1],
           ["Hint Okyanusu", "Indian Ocean", -25.0, 85.0, 0.1],
           ["Avustralya iç kesimleri (ücra kara)", "Australian outback (remote land)", -26.0, 134.0, 0.3],
           ["Kazak bozkırı (seyrek kara)", "Kazakh steppe (sparse land)", 48.0, 65.0, 5],
           ["Orta Avrupa (yoğun kara)", "Central Europe (dense land)", 48.0, 10.0, 120],
           ["Güney Asya deltası (çok yoğun)", "South Asian delta (very dense)", 24.0, 88.0, 1000]]
SIM_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
*{box-sizing:border-box}body{margin:0;background:#020617;color:#e2e8f0;font-family:'Inter','Segoe UI',sans-serif;font-size:12.5px}
.g{display:grid;grid-template-columns:1.5fr 1fr;grid-template-rows:470px 190px;gap:8px;padding:8px;height:auto}
.box{position:relative;border:1px solid #0b3a6b;border-radius:12px;overflow:hidden;background:#050b1a}canvas{width:100%;height:100%;display:block}
.hud{position:absolute;top:26px;left:10px;background:rgba(3,7,18,.88);border:1px solid #003366;border-radius:10px;padding:8px 12px;line-height:1.6;color:#38bdf8;z-index:3}
.zb{position:absolute;right:8px;top:8px;z-index:3}.zb button{width:28px;height:28px;margin-left:4px}
.ctl{display:flex;flex-wrap:wrap;gap:10px 16px;padding:8px 10px;border-bottom:1px solid #0b3a6b;background:#050b1a}
.ctl label{display:flex;flex-direction:column;gap:2px;color:#94a3b8;min-width:150px}.ctl b{color:#38bdf8;font-family:'Space Grotesk',sans-serif}.hud b,#st,#tt,.rk,.zc b{font-family:'Space Grotesk',sans-serif}
.zc{border-radius:10px;padding:8px 10px;margin-bottom:8px;border:1px solid}.zc p{margin:4px 0;color:#cbd5e1;line-height:1.45}.zc span{color:#fff;font-weight:600}
.zc.r{background:rgba(239,68,68,.12);border-color:#ef4444}.zc.a{background:rgba(245,158,11,.12);border-color:#f59e0b}.zc.g{background:rgba(16,185,129,.12);border-color:#10b981}
select,input[type=range]{width:100%;background:#0b1329;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:3px}
button{background:#fc3d21;color:#fff;border:0;border-radius:8px;padding:6px 12px;font-weight:bold;cursor:pointer;font-family:'Inter',sans-serif}
button.s{background:#0f1b33;border:1px solid #1e3a5f}
.side{grid-row:1/3;grid-column:2;overflow-y:auto;padding:10px}
.rk{padding:8px;border-radius:8px;font-weight:bold;margin-bottom:8px;text-align:center}.red{background:#7f1d1d;color:#fecaca}.amber{background:#78350f;color:#fde68a}.green{background:#064e3b;color:#a7f3d0}
.kv{display:flex;justify-content:space-between;border-bottom:1px solid #12315a;padding:4px 0}.kv span{color:#94a3b8}
.note{margin:6px 0;padding:6px 8px;background:#0e2433;border-left:3px solid #38bdf8;border-radius:6px}
table{width:100%;margin-top:8px;border-collapse:collapse}td{padding:3px 4px;border-bottom:1px solid #12315a;font-size:11px}
.bar{display:flex;align-items:center;gap:8px;padding:6px 10px}#scrub{flex:1}
@media(max-width:900px){.g{grid-template-columns:1fr;grid-template-rows:340px 300px 170px auto;height:auto}.side{grid-row:auto;grid-column:1}}
#zones{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:4px 8px 14px}#zones .zc{margin:0;font-size:13px}#zones .zc p{font-size:12.5px}@media(max-width:900px){#zones{grid-template-columns:1fr}}</style></head><body>
<div class="ctl">
<label><span data-i data-tr="Kütle" data-en="Mass"></span><span><b id="mv"></b> kg</span><input id="m" type="range" min="50" max="5000" step="10"></label>
<label><span data-i data-tr="Kesit alanı" data-en="Cross-section"></span><span><b id="arv"></b> m²</span><input id="ar" type="range" min="0.1" max="30" step="0.1"></label>
<label><span data-i data-tr="Giriş açısı (120 km)" data-en="Entry angle (120 km)"></span><span><b id="anv"></b> °</span><input id="an" type="range" min="0.5" max="45" step="0.1"></label>
<label><span data-i data-tr="Malzeme" data-en="Material"></span><select id="mt"></select></label>
<label><span data-i data-tr="Hedef yüzey / düşüş bölgesi" data-en="Target surface / impact region"></span><select id="rg"></select></label>
<label><span data-i data-tr="Düşüş bölgesi nüfus yoğunluğu" data-en="Impact-region population"></span><select id="pp"><option value="0.5">Ocean / Okyanus</option><option value="5">Remote / Ücra</option><option value="50">World avg / Dünya ort.</option><option value="300">Dense / Yoğun</option><option value="3000">Urban / Kent</option></select></label>
<label><span data-i data-tr="Oynatma süresi (sn)" data-en="Playback length (s)"></span><select id="sp"><option value="60">60</option><option value="30" selected>30</option><option value="15">15</option></select></label>
</div>
<div class="bar"><button id="pl">⏯</button><button id="rs" class="s">↺</button><input id="scrub" type="range" min="0" max="1000" value="0"><b id="tt">0 s</b></div>
<div class="g">
<div class="box"><canvas id="side"></canvas><div class="hud">STATUS: <span id="st">—</span><br>ALT <span id="a">--</span> km · V <span id="v">--</span> km/s<br>TEMP <span id="t">--</span> °C · DECEL <span id="g">--</span> g<br>FLUX <span id="q">--</span> MW/m² · HEAT <span id="e">--</span></div><div class="zb"><button id="zi" class="s">+</button><button id="zo" class="s">−</button></div></div>
<div class="box side"><div id="risk"></div></div>
<div class="box" style="grid-row:2"><canvas id="chart"></canvas></div>
</div>
<div class="g" style="grid-template-rows:300px;height:auto;min-height:0;margin-top:-4px"><div class="box" style="grid-column:1/3"><canvas id="map"></canvas></div></div>
<div id="zones"></div>
<script>const D=__DATA__,TR=D.lang==='tr',tx=(a,b)=>TR?a:b,$=id=>document.getElementById(id);
const RE=6371e3,MU=3.986004418e14,SB=5.670374e-8,REC=0.5,DEG=Math.PI/180;
const rho=h=>{h=Math.max(h,0);for(let i=D.atm.length-1;i>=0;i--){const a=D.atm[i];if(h>=a[0])return a[1]*Math.exp(-(h-a[0])/a[2]);}return 1.225};
const rng=s=>()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const gauss=r=>Math.sqrt(-2*Math.log(r()+1e-9))*Math.cos(6.2832*r());
function integ(s,m,A,cd,rn,eDem,hm,dt,pc){let{h,v,g,psi,t}=s,E=0,rows=[],nx=t,fate='lost';
 for(let n=0;n<400000;n++){const r=RE+h,rh=rho(h/1e3),dr=.5*rh*cd*A*v*v/m,gr=MU/(r*r),q=1.7415e-4*Math.sqrt(rh/rn)*v**3,Tk=Math.max(Math.pow(q/(.8*SB),.25),250);
  E+=q*A*hm*dt;
  if(t>=nx-1e-9){rows.push([t,h/1e3,v/1e3,g/DEG,dr/9.80665,q/1e6,Tk-273.15,E/m/1e6,psi*RE/1e3]);nx+=REC}
  if(E/m>=eDem*1e6||(pc&&.5*rh*v*v>=pc)){fate='demise';break}
  v=Math.max(v+(-dr+gr*Math.sin(g))*dt,25);g+=(gr/v-v/r)*Math.cos(g)*dt;h-=v*Math.sin(g)*dt;psi+=v*Math.cos(g)/r*dt;t+=dt;
  if(h<=0){fate='impact';break}}
 return{rows,fate,m,A,end:rows[rows.length-1]}}
const PARTS=[['Propellant tank (Ti)','Yakıt tankı (Ti)',.09,4.2,.55,1],['Reaction wheels / steel','Reaksiyon çarkı / çelik',.06,2.6,.5,1],
['Primary bus structure','Ana gövde yapısı',.30,0,1,1],['Solar array','Güneş paneli',.18,.45,3.2,1],['Electronics box','Elektronik kutu',.13,.6,.8,1],
['Antenna / boom','Anten / kol',.06,.5,2.4,1],['Small debris','Küçük parça',.18,.8,1.2,12]];
const STARS=Array.from({length:90},(_,i)=>{const r=rng(i+5);return[r(),r(),.3+.6*r()]});
let P={},S={},T=0,play=true,tb=0,last=performance.now(),flash=-1,zoom=1;
function sim(){
 const A=P.area,M=P.mass,r=rng(Math.round(M*7+A*131+P.angle*977)),E0=P.mat;
 const main=integ({h:120e3,v:7800,g:P.angle*DEG,psi:0,t:0},M,A,2.2,.5,E0,2,.05,430*Math.sqrt(E0/.97)*(1+.12*Math.log(M/1200)));
 let frags=[],xb=0,tbr=null;
 if(main.fate==='demise'){tbr=main.end;const e=main.end;
  const st={h:e[1]*1e3,v:e[2]*1e3,g:e[3]*DEG,psi:e[8]*1e3/RE,t:e[0]};tb=e[0];xb=e[8];
  PARTS.forEach((p,k)=>{const nn=p[5]>1?Math.round(p[5]*Math.max(1,Math.sqrt(M/400))):1;for(let j=0;j<nn;j++){const f=p[2]/nn*(.7+.6*r()),m=Math.max(M*f,.05),a=Math.max(A*Math.pow(f,.67)*p[4]*(.8+.4*r()),.002);
   const ed=(p[3]||E0)*(.8+.5*r()),cd=.7+1.6*r(),ff=integ({h:st.h,psi:st.psi,t:st.t,v:st.v+gauss(r)*28,g:Math.max(.003,st.g+gauss(r)*.012)},m,a,cd,.1+.3*Math.sqrt(a),ed,2,.1);ff.name=TR?p[1]:p[0];ff.k=k;frags.push(ff)}});
 }else{tb=-1;main.name=tx('Nesne (bütün)','Intact object');frags=[main]}
 const sv=frags.filter(f=>f.fate==='impact'),xs=sv.map(f=>f.end[8]);
 let xmin=xb,xmax=xb;if(sv.length){xmin=Math.min(...xs);xmax=Math.max(...xs)}
 const len=Math.max(xmax-xmin,5),w3=Math.max(6,.07*len),sg=w3/3;
 let ec=0;sv.forEach(f=>{f.ke=.5*f.m*Math.pow(f.end[2]*1e3,2);f.vi=f.end[2]*1e3;f.cx=(r()-.5)*0+gauss(r)*sg;f.x=f.end[8];
  f.ac=Math.pow(Math.sqrt(f.A)+.6,2);f.cas=f.ke>=15;if(f.cas)ec+=f.ac*P.pop/1e6});
 const pcv=430*Math.sqrt(E0/.97)*(1+.12*Math.log(M/1200));
 S={tp:(main.rows.find(x=>.5*rho(x[1])*Math.pow(x[2]*1e3,2)>=.45*pcv)||main.end)[0],main,frags,sv,xb,xmin,xmax,len,w3,ec,tmax:Math.max(main.end[0],...frags.map(f=>f.end[0]))+5};ORGset()}
let ORG={lat:0,lon:0};
function ORGset(){const rg=P.reg>=0?D.regions[P.reg]:null;if(rg){const c=dest(rg[2],rg[3],D.az+Math.PI,(S.xmin+S.xmax)/2);ORG={lat:c[0],lon:c[1]};for(let i=0;i<3;i++){const q=geo((S.xmin+S.xmax)/2,0);ORG={lat:ORG.lat+rg[2]-q[0],lon:ORG.lon+rg[3]-q[1]}}}else ORG={lat:D.lat,lon:D.lon}}
function at(rows,t,i){const f=Math.min(Math.max(t/REC,0),rows.length-1),a=Math.floor(f),b=Math.min(a+1,rows.length-1),u=f-a,ra=rows[a],rb=rows[b];return ra.map((x,k)=>x+(rb[k]-x)*u)[i]}
function row(rows,t){const f=Math.min(Math.max((t-rows[0][0])/REC,0),rows.length-1),a=Math.floor(f),b=Math.min(a+1,rows.length-1),u=f-a;return rows[a].map((x,k)=>x+(rows[b][k]-x)*u)}
function dest(la,lo,az,km){const d=km/6371,p=la*DEG,l=lo*DEG,b=az,p2=Math.asin(Math.sin(p)*Math.cos(d)+Math.cos(p)*Math.sin(d)*Math.cos(b)),l2=l+Math.atan2(Math.sin(b)*Math.sin(d)*Math.cos(p),Math.cos(d)-Math.sin(p)*Math.sin(p2));return[p2/DEG,l2/DEG]}
function geo(x,off){const c=dest(ORG.lat,ORG.lon,D.az,x);if(!off)return c;return dest(c[0],c[1],D.az+Math.PI/2,off)}
const U=(a,c)=>{while(a-c>180)a-=360;while(a-c<-180)a+=360;return a};
const cs=$('side'),cm=$('map'),cc=$('chart'),X=cs.getContext('2d'),M2=cm.getContext('2d'),C=cc.getContext('2d');
const tex=new Image();let texOK=false;tex.onload=()=>texOK=true;tex.src='https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg';
function fit(c){const r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);c.width=r.width*d;c.height=r.height*d;c.getContext('2d').setTransform(d,0,0,d,0,0);c.w=r.width;c.h=r.height}
function heatCol(t){return t>2000?'#fff3c0':t>1400?'#ffb347':t>800?'#ff6a1a':t>400?'#b33a1a':'#8a96a8'}
function sprite(c,x,y,sc,ang,hot,split){const b=Math.sqrt(P.area*.4)*sc,hg=Math.cbrt(P.mass/400)*sc*1.1,pw=Math.sqrt(P.area*.6)*sc,o=split*pw*1.6;
 c.save();c.translate(x,y);c.rotate(ang);
 [-1,1].forEach(sd=>{c.save();c.translate(sd*(b/2+pw*.75+o),sd*o*.5);c.rotate(sd*split*1.2);const w=pw*1.5,h=hg*.34,g=c.createLinearGradient(-w/2,0,w/2,0);g.addColorStop(0,'#1d4ed8');g.addColorStop(.5,'#3b82f6');g.addColorStop(1,'#1e3a8a');c.fillStyle=g;c.fillRect(-w/2,-h/2,w,h);c.strokeStyle='rgba(191,219,254,.6)';c.lineWidth=.6;for(let i=1;i<5;i++){c.beginPath();c.moveTo(-w/2+w*i/5,-h/2);c.lineTo(-w/2+w*i/5,h/2);c.stroke()}c.strokeRect(-w/2,-h/2,w,h);if(hot>.2){c.fillStyle=`rgba(255,120,30,${Math.min(.7,hot*.8)})`;c.fillRect(-w/2,-h/2,w,h)}c.restore()});
 const g=c.createLinearGradient(-b/2,-hg/2,b/2,hg/2);g.addColorStop(0,'#f2d27a');g.addColorStop(.5,'#c89b35');g.addColorStop(1,'#8a6420');c.fillStyle=g;c.fillRect(-b/2,-hg/2,b,hg);
 if(hot>.1){c.fillStyle=`rgba(255,${(170-hot*90)|0},40,${Math.min(.85,hot)})`;c.fillRect(-b/2,-hg/2,b,hg)}
 c.strokeStyle='#e2e8f0';c.lineWidth=1;c.strokeRect(-b/2,-hg/2,b,hg);c.beginPath();c.arc(0,-hg/2-b*.12,b*.3,Math.PI,0);c.fillStyle='#e5e7eb';c.fill();c.restore()}
let trailPts=[];
function drawSide(t){const W=cs.w,H=cs.h,ho=t<S.main.end[0]?row(S.main.rows,t):S.main.end,fx=ho[8],win=260/zoom,top=130,gy=H-34,sy=h=>gy-(h/top)*(gy-30),sx=x=>W/2+(x-fx)/win*W;
 X.save();if(S.tb>0&&Math.abs(t-S.tb)<1.2)X.translate((Math.random()-.5)*6,(Math.random()-.5)*6);
 const g=X.createLinearGradient(0,0,0,gy);g.addColorStop(0,'#000208');g.addColorStop(.45,'#030b22');g.addColorStop(.75,'#0b2a5e');g.addColorStop(.92,'#2f6db5');g.addColorStop(1,'#8ec5f0');X.fillStyle=g;X.fillRect(-10,-10,W+20,H+20);
 STARS.forEach(s=>{X.fillStyle=`rgba(255,255,255,${s[2]})`;X.fillRect(s[0]*W,s[1]*H*.45,1.4,1.4)});
 const R=W*2.6;X.save();X.beginPath();X.arc(W/2,gy+R,R,0,7);X.clip();const eg=X.createLinearGradient(0,gy,0,H);eg.addColorStop(0,'#2d6a4f');eg.addColorStop(1,'#0b2e1f');X.fillStyle=eg;X.fillRect(0,gy,W,H);
 X.fillStyle='rgba(255,255,255,.22)';for(let i=-6;i<24;i++){X.beginPath();X.ellipse(sx(i*48+((i*37)%19)),gy+8+((i*53)%17),26+(i*7)%18,4,0,0,7);X.fill()}X.restore();
 X.save();X.shadowColor='#7cc4ff';X.shadowBlur=22;X.strokeStyle='rgba(150,210,255,.85)';X.lineWidth=2.5;X.beginPath();X.arc(W/2,gy+R,R,Math.PI*1.2,Math.PI*1.8);X.stroke();X.restore();
 X.font='10px Inter,sans-serif';X.textAlign='left';[[100,'100 km · '+tx('Karman hattı','Kármán line')],[80,'80 km'],[60,'60 km'],[40,'40 km'],[20,'20 km']].forEach(a=>{const y=sy(a[0]);X.strokeStyle='rgba(148,163,184,.22)';X.setLineDash([4,5]);X.beginPath();X.moveTo(0,y);X.lineTo(W,y);X.stroke();X.setLineDash([]);X.fillStyle='#a8b6cc';X.fillText(a[1],6,y-3)});
 X.textAlign='right';X.fillStyle='rgba(203,213,225,.55)';[[110,tx('TERMOSFER','THERMOSPHERE')],[68,tx('MEZOSFER','MESOSPHERE')],[30,tx('STRATOSFER','STRATOSPHERE')]].forEach(a=>X.fillText(a[1],W-8,sy(a[0])));X.textAlign='left';
 X.save();X.strokeStyle='rgba(255,120,60,.5)';X.shadowColor='#ff5a1f';X.shadowBlur=8;X.lineWidth=1.6;X.beginPath();S.main.rows.forEach((p,i)=>{const x=sx(p[8]),y=sy(p[1]);i?X.lineTo(x,y):X.moveTo(x,y)});X.stroke();X.restore();
 if(t<=S.main.end[0]){const fl=Math.min(1,ho[5]/Math.max(S.pk,1e-9)),x=sx(ho[8]),y=sy(ho[1]),p2=row(S.main.rows,Math.min(t+1,S.main.end[0])),vx=sx(p2[8])-x,vy=sy(p2[1])-y,a=Math.atan2(vy||.001,vx||1),tl=40+240*fl,sc=Math.min(2.6,Math.max(.8,2.4/Math.sqrt(zoom)))*(H/520)*7;
  if(fl>.02){X.save();X.globalCompositeOperation='lighter';
   for(let k=0;k<46;k++){const u=k/46,px=x-Math.cos(a)*u*tl,py=y-Math.sin(a)*u*tl,r=(6+26*fl)*(1-u*.8)*(.9+.2*Math.sin(t*20+k)),al=fl*(1-u)*.5,gr=X.createRadialGradient(px,py,0,px,py,r);gr.addColorStop(0,`rgba(255,${(240-120*u)|0},${(200-170*u)|0},${al})`);gr.addColorStop(.5,`rgba(255,${(120-60*u)|0},40,${al*.6})`);gr.addColorStop(1,'rgba(120,40,200,0)');X.fillStyle=gr;X.beginPath();X.arc(px,py,r,0,7);X.fill()}
   const bg=X.createRadialGradient(x,y,0,x,y,22+50*fl);bg.addColorStop(0,`rgba(255,255,255,${.7*fl})`);bg.addColorStop(.4,`rgba(255,200,120,${.5*fl})`);bg.addColorStop(1,'rgba(255,90,20,0)');X.fillStyle=bg;X.beginPath();X.arc(x,y,22+50*fl,0,7);X.fill();
   const rr=rng(7);for(let i=0;i<34;i++){const ph=(t*.8+i*.173)%1,d=ph*tl*1.2,off=(rr()-.5)*40*fl*(ph+.2);X.fillStyle=`rgba(255,${(180+rr()*70)|0},90,${(1-ph)*fl})`;X.fillRect(x-Math.cos(a)*d-Math.sin(a)*off,y-Math.sin(a)*d+Math.cos(a)*off,2,2)}
   X.restore();X.strokeStyle=`rgba(255,160,200,${.5*fl})`;X.lineWidth=2;X.beginPath();X.arc(x,y,16+14*fl,a-1,a+1);X.stroke()}
  const sp=S.tb>0?Math.min(1,Math.max(0,(t-S.tp)/Math.max(S.tb-S.tp,1))):0;sprite(X,x,y,sc,a+Math.sin(t*.7)*.3+(sp>0?t*.9:0),fl,sp)}
 if(S.tb>0&&t>=S.tb){const age=t-S.tb;if(age<2.5){const x=sx(S.xb),y=sy(S.main.end[1]),rr=age*70;X.strokeStyle=`rgba(255,240,200,${1-age/2.5})`;X.lineWidth=3;X.beginPath();X.arc(x,y,rr,0,7);X.stroke();X.fillStyle=`rgba(255,200,120,${.8-age/3})`;X.beginPath();X.arc(x,y,rr*.5,0,7);X.fill()}}
 S.frags.forEach(f=>{if(S.tb>0&&t<S.tb)return;const r=f.rows,t1=r[r.length-1][0];if(S.tb<0)return;const p=row(r,Math.min(t,t1)),x=sx(p[8]),y=sy(p[1]),gone=f.fate==='demise'&&t>t1,fade=gone?Math.max(0,1-(t-t1)/3):1;if(fade<=0)return;
  const n=Math.min(8,r.length);X.strokeStyle=`rgba(255,150,60,${.35*fade})`;X.lineWidth=2;X.beginPath();for(let k=n;k>=0;k--){const q=row(r,Math.max(r[0][0],Math.min(t,t1)-k*1.2));k===n?X.moveTo(sx(q[8]),sy(q[1])):X.lineTo(sx(q[8]),sy(q[1]))}X.stroke();
  const rad=2+Math.min(5,Math.sqrt(f.m)*.6);X.fillStyle=gone?`rgba(255,200,120,${fade})`:heatCol(p[6]);X.shadowColor='#ff7a1a';X.shadowBlur=p[6]>600?14:0;X.beginPath();X.arc(x,y,rad,0,7);X.fill();X.shadowBlur=0;
  if(f.fate==='impact'&&t>=t1){X.fillStyle='#fde047';X.beginPath();X.arc(x,sy(0),5,0,7);X.fill()}});
 X.fillStyle='#cbd5e1';X.font='11px Inter,sans-serif';X.fillText(tx('YAN GÖRÜNÜM · irtifa dikey olarak abartılı','SIDE VIEW · altitude vertically exaggerated'),8,14);X.restore()}
function mapView(){const x0=S.tb>0?Math.min(S.xb,S.xmin):S.xmin,rg=Math.max(S.xmax-x0,20),c=geo((x0+S.xmax)/2,0),span=Math.max(rg*1.7/111/Math.max(Math.cos(c[0]*DEG),.2),10);return{c,span,a:x0-.2*rg,b:S.xmax+.3*rg}}
function drawMap(t){const W=cm.w,H=cm.h,v=mapView(),sp=v.span,spv=sp*H/W,c=v.c,lc=c[1];
 const px=(la,lo)=>[(U(lo,lc)-lc)/sp*W+W/2,(c[0]-la)/spv*H+H/2];M2.fillStyle='#07162e';M2.fillRect(0,0,W,H);
 if(texOK){const iw=tex.width,ih=tex.height,sx=(lc-sp/2+180)/360*iw,sw=sp/360*iw,sy=(90-c[0]-spv/2)/180*ih,sh=spv/180*ih;try{M2.drawImage(tex,sx,Math.max(0,sy),sw,Math.min(sh,ih-Math.max(0,sy)),0,0,W,H)}catch(e){}}
 M2.fillStyle='rgba(16,185,129,.20)';M2.fillRect(0,0,W,H);M2.strokeStyle='rgba(255,255,255,.12)';M2.lineWidth=1;M2.font='10px Inter,sans-serif';
 const st=sp>40?10:sp>15?5:2;for(let la=Math.ceil((c[0]-spv/2)/st)*st;la<c[0]+spv/2;la+=st){const y=px(la,lc)[1];M2.beginPath();M2.moveTo(0,y);M2.lineTo(W,y);M2.stroke();M2.fillStyle='#94a3b8';M2.fillText(la+'°',3,y-2)}
 for(let lo=Math.ceil((lc-sp/2)/st)*st;lo<lc+sp/2;lo+=st){const x=px(c[0],lo)[0];M2.beginPath();M2.moveTo(x,0);M2.lineTo(x,H);M2.stroke();M2.fillText(((lo+540)%360-180)+'°',x+2,H-3)}
 const zone=(k,col,bor)=>{const a=S.xmin-k*.12*S.len-(k>1?20:0),b=S.xmax+k*.12*S.len+(k>1?20:0),w=S.w3*k,pts=[];for(let i=0;i<=16;i++)pts.push(geo(a+(b-a)*i/16,w));for(let i=16;i>=0;i--)pts.push(geo(a+(b-a)*i/16,-w));
  M2.beginPath();pts.forEach((p,i)=>{const q=px(p[0],p[1]);i?M2.lineTo(q[0],q[1]):M2.moveTo(q[0],q[1])});M2.closePath();M2.fillStyle=col;M2.fill();M2.strokeStyle=bor;M2.lineWidth=1.5;M2.stroke()};
 zone(2.2,'rgba(245,158,11,.30)','#f59e0b');zone(1,'rgba(239,68,68,.45)','#ef4444');
 {const mc=(S.xmin+S.xmax)/2;M2.font='bold 12px Inter,sans-serif';M2.textAlign='center';[[0,'#fecaca',tx('KIRMIZI · TEHLİKE','RED · HAZARD')],[S.w3*1.9,'#fde68a',tx('SARI · DİKKAT','AMBER · CAUTION')]].forEach(z=>{const p=geo(mc,z[0]),q=px(p[0],p[1]);M2.fillStyle=z[1];M2.fillText(z[2],q[0],q[1]+4)});M2.textAlign='left'}
 M2.setLineDash([6,5]);M2.strokeStyle='#e2e8f0';M2.beginPath();for(let i=0;i<=40;i++){const p=geo(v.a+(v.b-v.a)*i/40,0),q=px(p[0],p[1]);i?M2.lineTo(q[0],q[1]):M2.moveTo(q[0],q[1])}M2.stroke();M2.setLineDash([]);
 S.frags.forEach(f=>{const r=f.rows,t1=r[r.length-1][0];if(S.tb>0&&t<S.tb)return;const live=t<t1;if(!live&&f.fate!=='impact')return;
  const xx=live?row(r,t)[8]:f.x,cx=f.cx?f.cx*(live?Math.min(1,Math.max(0,(t-S.tb)/Math.max(t1-S.tb,1))):1):0,p=geo(xx,cx),q=px(p[0],p[1]);
  if(live){M2.fillStyle='rgba(255,150,60,.9)';M2.beginPath();M2.arc(q[0],q[1],2,0,7);M2.fill()}else{const age=t-t1;if(age<4){M2.strokeStyle=`rgba(253,224,71,${1-age/4})`;M2.lineWidth=2;M2.beginPath();M2.arc(q[0],q[1],4+age*9,0,7);M2.stroke()}M2.fillStyle=f.cas?'#fde047':'#fff';M2.beginPath();M2.arc(q[0],q[1],2+Math.min(4,Math.sqrt(f.m)*.4),0,7);M2.fill()}});
 if(S.tb>0){const p=geo(S.xb,0),q=px(p[0],p[1]);M2.strokeStyle='#fc3d21';M2.lineWidth=2;M2.beginPath();M2.arc(q[0],q[1],7,0,7);M2.stroke();M2.fillStyle='#fff';M2.fillText('✹ '+tx('PARÇALANMA','BREAK-UP'),q[0]+10,q[1]-6)}
 const ho=t<S.main.end[0]?row(S.main.rows,t):S.main.end,p=geo(ho[8],0),q=px(p[0],p[1]);M2.fillStyle='#38bdf8';M2.beginPath();M2.arc(q[0],q[1],5,0,7);M2.fill();
 M2.font='bold 11px Inter,sans-serif';M2.fillStyle='#ef4444';M2.fillText('■ '+tx('KIRMIZI BÖLGE (yasak/tehlike)','RED ZONE (keep-out)'),8,14);M2.fillStyle='#f59e0b';M2.fillText('■ '+tx('SARI: dikkat / NOTAM alanı','AMBER: caution / NOTAM area'),8,28);M2.fillStyle='#10b981';M2.fillText('■ '+tx('YEŞİL: güvenli bölge','GREEN: safe zone'),8,42)}
function drawChart(t){const W=cc.w,H=cc.h,r=S.main.rows;C.fillStyle='#050b1a';C.fillRect(0,0,W,H);const T0=r[r.length-1][0]||1;
 const ser=[[5,'#fc3d21',S.pk,tx('Isı akısı','Heat flux')],[6,'#f59e0b',Math.max(...r.map(a=>a[6]),1),tx('Sıcaklık','Temp')],[4,'#38bdf8',Math.max(...r.map(a=>a[4]),1),tx('Yavaşlama g','Decel g')],[7,'#a78bfa',P.mat,tx('Birikmiş ısı / eşik','Heat load / limit')]];
 ser.forEach((s,k)=>{C.strokeStyle=s[1];C.lineWidth=1.6;C.beginPath();r.forEach((p,i)=>{const x=p[0]/T0*(W-10)+5,y=H-8-(p[s[0]]/s[2])*(H-24);i?C.lineTo(x,y):C.moveTo(x,y)});C.stroke();C.fillStyle=s[1];C.font='10px Inter,sans-serif';C.fillText(s[3],8+k*105,12)});
 const cx=Math.min(t,T0)/T0*(W-10)+5;C.strokeStyle='#fff';C.beginPath();C.moveTo(cx,0);C.lineTo(cx,H);C.stroke();if(S.tb>0){const bx=S.tb/T0*(W-10)+5;C.strokeStyle='#fc3d21';C.setLineDash([3,3]);C.beginPath();C.moveTo(bx,0);C.lineTo(bx,H);C.stroke();C.setLineDash([])}}
function hud(t){const ho=t<S.main.end[0]?row(S.main.rows,t):S.main.end;let s=tx('ALÇALMA','DESCENT'),c='#38bdf8';
 if(S.tb>0&&t>=S.tb){s=tx('💥 PARÇALANMA / ENKAZ BULUTU','💥 BREAK-UP / DEBRIS CLOUD');c='#fc3d21'}else if(S.tb>0&&t>=S.tp){s=tx('⚠ PANEL KOPMASI / YAPISAL ZORLANMA','⚠ APPENDAGE SHEDDING / STRUCTURAL LOAD');c='#f59e0b'}else if(ho[5]/Math.max(S.pk,1e-9)>.1){s=tx('🔥 PLAZMA ISINMASI','🔥 PLASMA HEATING');c='#f97316'}
 $('st').textContent=s;$('st').style.color=c;$('a').textContent=ho[1].toFixed(1);$('v').textContent=ho[2].toFixed(2);$('t').textContent=ho[6].toFixed(0);$('g').textContent=ho[4].toFixed(1);$('q').textContent=ho[5].toFixed(2);$('e').textContent=(ho[7]/P.mat*100).toFixed(0)+'%';$('tt').textContent=t.toFixed(0)+' s';$('scrub').value=t/S.tmax*1000}
function zoneArea(k){const e=k*.12*S.len+(k>1?20:0);return(S.len+2*e)*(2*S.w3*k)}
function legend(){const ar=zoneArea(1),aa=zoneArea(2.2)-ar,pe=Math.round(ar*P.pop);
 return`<div class="zc r"><b>🔴 ${tx('KIRMIZI BÖLGE · Tehlike / Yasak alan','RED ZONE · Hazard / keep-out')}</b><p>${tx('Yere ulaşan ve ≥15 J enerjili parçaların (can kaybı eşiği) %99,7 olasılıkla ineceği koridor (±3σ). Deniz ve hava trafiğine kapatılır, insanlar tahliye/barınma uyarısı alır.','Corridor where surviving fragments with ≥15 J (casualty threshold) land with 99.7% probability (±3σ). Closed to ships and aircraft; people are told to shelter or evacuate.')}</p><span>${ar.toFixed(0)} km² · ~${pe.toLocaleString()} ${tx('kişi (seçili yoğunlukta)','people (at selected density)')}</span></div>
 <div class="zc a"><b>🟡 ${tx('SARI BÖLGE · Dikkat / NOTAM alanı','AMBER ZONE · Caution / NOTAM area')}</b><p>${tx('Rüzgâr, takla ve parçalanma irtifası belirsizliğini kapsayan tampon. Havacılık (NOTAM) ve denizcilik (NAVAREA) uyarısı yayınlanır; geçiş kısıtlanabilir.','Buffer covering wind, tumbling and break-up altitude uncertainty. Aviation (NOTAM) and maritime (NAVAREA) warnings are issued; transit may be restricted.')}</p><span>+${aa.toFixed(0)} km²</span></div>
 <div class="zc g"><b>🟢 ${tx('YEŞİL · Güvenli bölge','GREEN · Safe zone')}</b><p>${tx('Haritanın geri kalanı: parça düşme olasılığı ihmal edilebilir, önlem gerekmez.','Rest of the map: fragment-fall probability is negligible; no action needed.')}</p></div>`}
function riskUI(){const ec=S.ec,g=ec>1e-4?['red',tx('KIRMIZI · NASA 1:10.000 sınırı AŞILDI','RED · exceeds NASA 1:10,000 limit')]:ec>1e-5?['amber',tx('SARI · sınıra yakın','AMBER · near limit')]:['green',tx('YEŞİL · kabul edilebilir','GREEN · acceptable')];
 $('zones').innerHTML=legend();let h=`<div class="rk ${g[0]}">${g[1]}</div><div class="kv"><span>${tx('Beklenen kayıp (Ec)','Casualty expectation (Ec)')}</span><b>${ec>0?'1 : '+Math.round(1/ec).toLocaleString():'0'}</b></div>
 <div class="kv"><span>${tx('Parçalanma irtifası','Break-up altitude')}</span><b>${S.tb>0?S.main.end[1].toFixed(1)+' km':tx('yok (bütün iniş)','none (intact)')}</b></div>
 <div class="kv"><span>${tx('Hayatta kalan / toplam','Survivors / total')}</span><b>${S.sv.length} / ${S.frags.length}</b></div>
 <div class="kv"><span>${tx('Kırmızı bölge','Red zone')}</span><b>${(S.len+S.len*.24).toFixed(0)} × ${(S.w3*2).toFixed(0)} km</b></div>
 <div class="kv"><span>${tx('Hedef yörünge','Target orbit')}</span><b>${D.alt} km · ${D.inc}°</b></div>`;
if(P.reg>=0){const r=D.regions[P.reg];h+=`<div class="kv"><span>${tx('Hedef yüzey','Target surface')}</span><b>${TR?r[0]:r[1]}</b></div><div class="kv"><span>${tx('Nüfus yoğunluğu','Population density')}</span><b>${r[4]}/km²</b></div>`;if(Math.abs(r[2])>D.inc+1)h+=`<div class="note">⚠ ${tx('Yörünge eğimi bu enleme ulaşmıyor; kontrollü yeniden giriş için düzlem değiştirme manevrası gerekir.','Orbit inclination does not reach this latitude; a plane-change burn would be required for a controlled re-entry.')}</div>`}
 if(D.risk.n!=null){h+=`<div class="kv"><span>${tx('Çarpışma taraması','Conjunction scan')}</span><b>${D.risk.n} ${tx('yakın geçiş','close appr.')}</b></div>`;
  h+=D.risk.hit?`<div class="note">⚠ ${tx('Bu nesne taramada','This object is in scan')}: ${D.risk.hit.other} · ${D.risk.hit.miss} km · ${D.risk.hit.risk}</div>`:`<div class="note">${tx('Bu nesne için yakın geçiş kaydı yok.','No close approach on record for this object.')}</div>`}
 else h+=`<div class="note">${tx('Çarpışma Riski sayfasında tarama çalıştırılırsa sonuçlar burada görünür.','Run the scan on the Collision Risk page to link results here.')}</div>`;
 if(D.defense)h+=`<div class="note">🛡 ${tx('Savunma matrisi aktif','Defense matrix armed')}</div>`;
 const top=[...S.frags].sort((a,b)=>(b.ke||0)-(a.ke||0)).slice(0,6);
 h+='<table>'+top.map(f=>`<tr><td>${f.name}</td><td>${f.m.toFixed(1)} kg</td><td>${f.fate==='impact'?(f.vi.toFixed(0)+' m/s · '+(f.ke/1e3).toFixed(1)+' kJ'):'🔥 '+f.end[1].toFixed(0)+' km'}</td></tr>`).join('')+'</table>';$('risk').innerHTML=h}
function recompute(){P={mass:+$('m').value,area:+$('ar').value,angle:+$('an').value,mat:+$('mt').value,pop:+$('pp').value,reg:+$('rg').value};if(P.reg>=0)P.pop=D.regions[P.reg][4];$('pp').disabled=P.reg>=0;
 ['m','ar','an'].forEach(i=>$(i+'v').textContent=$(i).value);sim();S.pk=Math.max(...S.main.rows.map(a=>a[5]),1e-9);S.tb=tb;T=0;flash=-1;riskUI();play=true}
function frame(now){requestAnimationFrame(frame);const dt=Math.min((now-last)/1000,.1);last=now;
 if(play){const base=Math.max(1,S.tmax/(+$('sp').value)),nb=S.tb>0&&Math.abs(T-S.tb)<10?.18:1;T+=dt*base*nb;if(T>=S.tmax){T=S.tmax;play=false}}
 drawSide(T);drawMap(T);drawChart(T);hud(T)}
function init(){const mats=D.mats;$('mt').innerHTML=mats.map(m=>`<option value="${m[1]}">${TR?m[2]:m[0]}</option>`).join('');
 $('m').value=D.mass;$('ar').value=D.area;$('an').value=D.angle;$('mt').value=D.mat;$('pp').value=D.pop;
 document.querySelectorAll('[data-i]').forEach(e=>e.textContent=tx(e.dataset.tr,e.dataset.en));
 $('rg').innerHTML=`<option value="-1">${tx('Doğal (TLE yer izi)','Natural (TLE ground track)')}</option>`+D.regions.map((r,i)=>`<option value="${i}">${TR?r[0]:r[1]}</option>`).join('');$('rg').value=-1;['m','ar','an','mt','pp','rg'].forEach(i=>$(i).oninput=recompute);$('pl').onclick=()=>{if(T>=S.tmax)T=0;play=!play};$('rs').onclick=()=>{T=0;play=true};
 $('scrub').oninput=e=>{T=e.target.value/1000*S.tmax;play=false};$('zi').onclick=()=>zoom=Math.min(zoom*1.6,8);$('zo').onclick=()=>zoom=Math.max(zoom/1.6,.5);
 [cs,cm,cc].forEach(fit);addEventListener('resize',()=>[cs,cm,cc].forEach(fit));recompute();requestAnimationFrame(frame)}
init();
</script></body></html>
"""


def _mat_name(m):
    return _MAT_TR.get(m, m) if st.session_state.lang == "tr" else m


def _sim_open():
    st.session_state.sg_sim_cfg = {k: st.session_state.get(s) for k, s in
                                   (("target", "sg_target"), ("mass", "sg_mass"), ("area", "sg_area"),
                                    ("angle", "sg_angle"), ("material", "sg_material"))}
    st.session_state.sg_sim_mode = True


def _sim_close():
    st.session_state.sg_sim_mode = False


def _sim_orbit(name):
    """Hedef uydunun TLE'sinden şu anki irtifa/eğim/alt-nokta ve yer izi yönü (SGP4)."""
    out = {"alt": 420, "inc": 51.6, "lat": 0.0, "lon": 0.0, "az": math.asin(math.cos(math.radians(51.6)))}
    try:
        for n, a, b in CAT["groups"]["visual"] + CAT["groups"]["active"]:
            if n != name:
                continue
            s = Satrec.twoline2rv(a, b)
            jd, fr = to_jd(utc_now())
            e, r, v = s.sgp4(jd, fr)
            if e != 0:
                break
            rn = math.sqrt(sum(c * c for c in r))
            gmst = math.radians((280.46061837 + 360.98564736629 * (jd + fr - 2451545.0)) % 360)
            lat = math.asin(r[2] / rn)
            inc = s.inclo
            az = math.asin(max(-1, min(1, math.cos(inc) / max(math.cos(lat), 1e-6))))
            if v[2] < 0:
                az = math.pi - az
            out.update(alt=round(rn - RE), inc=round(math.degrees(inc), 2), lat=math.degrees(lat),
                       lon=(math.degrees(math.atan2(r[1], r[0]) - gmst) + 540) % 360 - 180, az=az)
            break
    except Exception:
        pass
    return out


def _sim_risk(name):
    """Çarpışma Riski sayfasındaki tarama sonuçlarından bu nesneye ait kaydı bağlar."""
    df, meta = st.session_state.get("conj_df"), st.session_state.get("conj_meta")
    res = {"n": None, "hit": None}
    try:
        if df is not None:
            res["n"] = int(len(df))
            if not df.empty and name != "Custom object":
                m = df[(df["Object A"] == name) | (df["Object B"] == name)]
                if not m.empty:
                    row = m.sort_values("Miss Distance (km)").iloc[0]
                    res["hit"] = {"other": str(row["Object B"] if row["Object A"] == name else row["Object A"]),
                                  "miss": round(float(row["Miss Distance (km)"]), 2), "risk": str(row.get("Risk", ""))}
        elif meta:
            res["n"] = int(meta.get("count", 0))
    except Exception:
        pass
    return res


if st.session_state.get("sg_sim_mode"):
    _cfg = st.session_state.get("sg_sim_cfg") or {}
    _tg = _cfg.get("target", "Custom object")
    _o = _sim_orbit(_tg)
    _payload = {"lang": st.session_state.lang, "atm": ATM, "mats": [[k, v, _MAT_TR[k]] for k, v in MATERIALS.items()],
                "mass": _cfg.get("mass", 1200), "area": _cfg.get("area", 4.0), "angle": _cfg.get("angle", 3.0),
                "mat": MATERIALS.get(_cfg.get("material"), 0.97), "pop": 50, "risk": _sim_risk(_tg),
                "regions": REGIONS, "defense": bool(st.session_state.get("defense_active")), **_o}
    st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');.block-container{padding-top:1rem !important}.stApp,.stApp p,.stApp button,.stApp label{font-family:'Inter','Segoe UI',sans-serif !important}.stApp h1,.stApp h2,.stApp h3{font-family:'Space Grotesk','Inter',sans-serif !important;letter-spacing:-.01em}</style>", unsafe_allow_html=True)
    _h1, _h2 = st.columns([5, 1])
    _tgt_lbl = T("Özel nesne", "Custom object") if _tg == "Custom object" else _tg
    _h1.markdown(f"### 🔥 SPACE GRAVE [SP4C3 GR4V3] {T('Yeniden Giriş ve Parçalanma Simülatörü', 'Re-Entry & Break-Up Simulator')} · <span style='color:#38bdf8'>{html_escape(str(_tgt_lbl))}</span>", unsafe_allow_html=True)
    _h2.button(T("✕ Simülasyondan çık", "✕ Exit simulation"), key="sg_sim_exit", on_click=_sim_close, **stretch(st.button))
    st.caption(T("Odak modu: site modülleri duraklatıldı; simülasyon tarayıcıda çalışır. Çıkınca her şey kaldığı yerden devam eder.",
                 "Focus mode: site modules are paused; the simulation runs in the browser. Everything resumes as you left it on exit."))
    embed_html(SIM_HTML.replace("__DATA__", json.dumps(_payload)), height=1400)
    st.stop()


_feed_word = FEED_LABEL.split(" ")[0]
st.markdown(_flat(f"""
    <div class="sg-mnav">
      <span class="sg-mlogo"><span class="sg-logo-mark"></span>SPACE GRAVE</span>
      <span class="sg-mlive"><span class="sg-dot{'' if _feed_word == 'LIVE' else ' warn'}"></span>{_feed_word}</span>
    </div>
"""), unsafe_allow_html=True)
with _keyed("sg_back"):
    st.button(T("← Giriş Sayfası", "← Home Page"), key="sg_back_btn", on_click=_back_to_intro)

_fw = FEED_LABEL.split(" ")[0]
_feed_chip = {"LIVE": ("c", T("CANLI TLE VERİSİ AKTİF", "LIVE TLE DATA ACTIVE")), "CACHED": ("a", T("TLE ÖNBELLEĞİ AKTİF", "TLE CACHE ACTIVE")),
              "PARTIAL": ("a", T("KISMİ TLE VERİSİ", "PARTIAL TLE DATA"))}.get(_fw, ("r", T("TLE ÇEVRİMDIŞI · YEDEK", "TLE OFFLINE · FALLBACK")))
st.markdown(_flat(f"""
<div class="ops-banner">
  <div>
    <div class="ops-eyebrow">NASA Space Apps Challenge · JPL {T("Komuta Merkezi", "Command Center")}</div>
    <div class="ops-title">SPACE GRAVE <span>[SP4C3 GR4V3]</span></div>
    <div class="ops-sub">{T("Küresel Uzay Çöpü Merkezi", "Global Space Debris Center")}</div>
  </div>
  <div class="ops-leds">
    <span class="ops-led g">SYSTEM ONLINE</span>
    <span class="ops-led {_feed_chip[0]}">{_feed_chip[1]}</span>
    <span class="ops-led c">SGP4 {T("MOTORU NORMAL", "PROPAGATOR NOMINAL")}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

PAGES = ["📊 General Status Overview", "🌍 3D Earth & Orbital Radar",
         "🚨 Collision Risk Analysis", "🔥 Re-Entry & Burn-Up Sim"]
_PAGES_TR = {
    PAGES[0]: "📊 Genel Durum Özeti", PAGES[1]: "🌍 3D Dünya ve Yörünge Radarı",
    PAGES[2]: "🚨 Çarpışma Riski Analizi", PAGES[3]: "🔥 Yeniden Giriş ve Yanma Sim.",
}


def page_label(p):
    return _PAGES_TR.get(p, p) if st.session_state.lang == "tr" else p


def _defense_txt():
    return T("AKTİF 🛡️", "ARMED 🛡️") if st.session_state.defense_active else T("BEKLEMEDE", "STANDBY")


def go_to(page):
    st.session_state.active_page = page


def _ops_ticker():
    n_a, n_d = len(CAT["groups"]["active"]), len(CAT["debris"])
    cj = st.session_state.conj_meta
    iss = None
    try:
        _m = df_api[df_api["Satellite Name"].str.upper().str.startswith("ISS")]
        iss = _m.iloc[0] if not _m.empty else None
        n_dec = int((df_api["Status"].str.startswith("DECAYING")).sum()) if not df_api.empty else 0
    except Exception:
        n_dec = 0
    it = [(T("KATALOG SENKRON", "CATALOG SYNC"), CAT["fetched"], ""),
          (T("TAKİPTEKİ NESNE", "OBJECTS TRACKED"), f"{n_a + n_d:,}", ""),
          (T("AKTİF UYDU", "ACTIVE SATS"), f"{n_a:,}", ""),
          (T("ENKAZ", "DEBRIS"), f"{n_d:,}", "r")]
    if iss is not None:
        it += [("ISS " + T("İRTİFA", "ALT"), f"{iss['Altitude (km)']} km", ""), ("ISS " + T("HIZ", "VEL"), f"{iss['Speed (km/s)']} km/s", ""),
               ("ISS " + T("EĞİM", "INC"), f"{iss['Inclination (°)']}°", "")]
    it += [(T("ALÇALAN (<300 km)", "DECAYING (<300 km)"), str(n_dec), "a" if n_dec else ""),
           (T("YAKIN GEÇİŞ", "CONJUNCTIONS"), f"{cj['count']}" if cj else "—", "a" if cj else ""),
           (T("SAVUNMA MATRİSİ", "DEFENSE MATRIX"), _defense_txt(), "g" if st.session_state.defense_active else ""),
           (T("YÖRÜNGE MOTORU", "PROPAGATOR"), "SGP4 NOMINAL", "g")]
    items = "".join(f'<span class="it"><i>{html_escape(a)}</i><b class="{c}">{html_escape(str(b))}</b></span><span class="sep">//</span>' for a, b, c in it)
    onl = {"LIVE": ("c", T("CANLI TLE VERİSİ AKTİF", "LIVE TLE DATA ACTIVE")), "CACHED": ("a", T("TLE ÖNBELLEĞİ AKTİF", "TLE CACHE ACTIVE")),
           "PARTIAL": ("a", T("KISMİ TLE VERİSİ", "PARTIAL TLE DATA"))}.get(FEED_LABEL.split(" ")[0], ("r", T("TLE ÇEVRİMDIŞI · YEDEK", "TLE OFFLINE · FALLBACK")))
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;font-family:'JetBrains Mono',ui-monospace,Consolas,monospace}
.t{display:flex;align-items:center;height:34px;background:rgba(5,11,20,.95);border-bottom:1px solid rgba(34,211,238,.18);border-top:1px solid rgba(34,211,238,.08);color:#9fb3cc;font-size:11px}
.lbl{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:0 14px;height:100%;border-right:1px solid rgba(34,211,238,.16);letter-spacing:.14em;font-weight:700;font-size:10.5px;white-space:nowrap}
.lbl::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--c);box-shadow:0 0 8px var(--c);animation:p 2s infinite}
.g{--c:#34d399;color:#34d399}.c{--c:#22d3ee;color:#22d3ee}.a{--c:#f59e0b;color:#f59e0b}.r{--c:#ef4444;color:#ef4444}
@keyframes p{50%{opacity:.3}}
.track{flex:1;min-width:0;overflow:hidden;-webkit-mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent);mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent)}
.run{display:inline-flex;align-items:center;white-space:nowrap;animation:m 80s linear infinite;will-change:transform}
.run:hover{animation-play-state:paused}
@keyframes m{to{transform:translateX(-50%)}}
.it{display:inline-flex;gap:8px;align-items:baseline;padding:0 4px}
.it i{font-style:normal;font-size:9.5px;letter-spacing:.16em;color:#6b7f9c}
.it b{font-weight:500;color:#67e8f9}.it b.r{color:#ef4444}.it b.a{color:#f59e0b}.it b.g{color:#34d399}
.sep{color:#1f3b57;padding:0 14px}
.clk{flex:0 0 auto;padding:0 14px;height:100%;display:flex;align-items:center;gap:8px;border-left:1px solid rgba(34,211,238,.16);color:#e2e8f0;font-weight:500;font-variant-numeric:tabular-nums;white-space:nowrap}
.clk i{font-style:normal;font-size:9.5px;letter-spacing:.16em;color:#6b7f9c}
@media(max-width:760px){.lbl.hide{display:none}}
</style></head><body><div class="t">
<div class="lbl g">SYSTEM ONLINE</div><div class="lbl hide __OC__">__OT__</div>
<div class="track"><div class="run">__IT____IT__</div></div>
<div class="clk"><i>UTC</i><span id="u">--:--:--</span></div></div>
<script>function t(){var d=new Date();document.getElementById('u').textContent=d.toISOString().slice(0,10)+' '+d.toISOString().slice(11,19);}t();setInterval(t,1000);</script>
</body></html>""".replace("__OC__", onl[0]).replace("__OT__", html_escape(onl[1])).replace("__IT__", items)
    with _keyed("sg_ticker"):
        embed_html(html, height=34)


_ops_ticker()

# on_click ile sayfa tek çalıştırmada değişir (st.rerun ile script iki kez çalışıyordu)
for _i, (col, page) in enumerate(zip(st.columns(len(PAGES)), PAGES)):
    with col:
        st.button(page_label(page), key=f"sg_nav_{_i}", on_click=go_to, args=(page,), **stretch(st.button))

_act_i = PAGES.index(st.session_state.active_page) if st.session_state.active_page in PAGES else 0
st.markdown(f"""<style>
.st-key-sg_nav_{_act_i} .stButton > button {{ background:rgba(34,211,238,.13) !important; border-color:#22d3ee !important; box-shadow:inset 0 -2px 0 #22d3ee, 0 0 20px rgba(34,211,238,.16) !important; }}
.st-key-sg_nav_{_act_i} .stButton > button p {{ color:#fff !important; }}
</style>""", unsafe_allow_html=True)

_lbl_mod, _lbl_feed, _lbl_upd = T("Aktif Modül", "Active Module"), T("Veri", "Feed"), T("Güncelleme", "Updated")
st.markdown(_flat(f"""<div class="ops-strip">
<span><i>{_lbl_mod}</i><b>{html_escape(page_label(st.session_state.active_page))}</b></span>
<span><i>{_lbl_feed}</i><b>{html_escape(FEED_LABEL)}</b></span>
<span><i>{_lbl_upd}</i><b>{CAT['fetched']}</b></span>
</div>"""), unsafe_allow_html=True)

# =====================================================================
# 1. MODÜL: OVERVIEW
# =====================================================================
if st.session_state.active_page == PAGES[0]:
    n_active = len(CAT["groups"]["active"])
    n_debris = len(CAT["debris"])
    conj = st.session_state.conj_meta
    conj_txt = f"{conj['count']}" if conj else "—"

    c = st.columns(6)
    cards = [
        ("nasa-glass-card", T("Katalog Nesneleri", "Catalog Objects"), f"{n_active + n_debris:,}", "metric-value", ""),
        ("nasa-glass-card", T("Enkaz Parçaları", "Debris Fragments"), f"{n_debris:,}", "metric-value", "color:#fc3d21;"),
        ("nasa-glass-card", T("Aktif Uydular", "Active Sats"), f"{n_active:,}", "metric-value", ""),
        ("nasa-glass-card fin-card", T("Yakın Geçişler", "Close Approaches"), conj_txt, "metric-value-fin", "color:#f59e0b;"),
        ("nasa-glass-card fin-card", T("Veri Durumu", "Feed Status"), FEED_LABEL, "metric-value-fin", "font-size:16px;"),
        ("nasa-glass-card", T("Savunma Matrisi", "Defense Matrix"), _defense_txt(), "metric-value", "color:#10b981;font-size:20px;"),
    ]
    for col, (cls, title, val, vcls, style) in zip(c, cards):
        with col:
            st.markdown(f'<div class="{cls}"><div class="metric-title">{title}</div><div class="{vcls}" style="{style}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown(T("#### 🛰️ Yörünge Kabukları: İrtifa × Eğim", "#### 🛰️ Orbital Shells: Altitude × Inclination"))
    rng = np.random.default_rng(3)

    def sample(alt, inc, label, n):
        if len(alt) == 0:
            return pd.DataFrame()
        idx = rng.choice(len(alt), min(n, len(alt)), replace=False)
        return pd.DataFrame({"Altitude (km)": alt[idx], "Inclination (°)": inc[idx], "Class": label})

    shells = pd.concat([sample(CAT["a_alt"], CAT["a_inc"], "Active", 3500),
                        sample(CAT["d_alt"], CAT["d_inc"], "Debris", 3500)])
    if not shells.empty:
        shells = shells[shells["Altitude (km)"] < 2200]
        _xl, _yl, _cl = T("Eğim (°)", "Inclination (°)"), T("İrtifa (km)", "Altitude (km)"), T("Sınıf", "Class")
        _lab = {"Active": T("Aktif", "Active"), "Debris": T("Enkaz", "Debris")}
        shells = shells.rename(columns={"Inclination (°)": _xl, "Altitude (km)": _yl, "Class": _cl})
        shells[_cl] = shells[_cl].map(_lab)
        fig = go.Figure()
        _cmap = {_lab["Active"]: "34,211,238", _lab["Debris"]: "239,68,68"}
        _hv = f"<b>%{{fullData.name}}</b><br>{_xl}: %{{x:.2f}}<br>{_yl}: %{{y:.0f}}<extra></extra>"
        for _nm, _g in shells.groupby(_cl):
            _rgb = _cmap.get(_nm, "148,163,184")
            # parlama (glow): iki yarı saydam halka + parlak çekirdek
            for _sz, _al in ((16, 0.035), (9, 0.09)):
                fig.add_trace(go.Scattergl(x=_g[_xl], y=_g[_yl], mode="markers", hoverinfo="skip", showlegend=False,
                                           marker=dict(size=_sz, color=f"rgba({_rgb},{_al})")))
            fig.add_trace(go.Scattergl(x=_g[_xl], y=_g[_yl], mode="markers", name=f"{_nm} · {len(_g):,}", hovertemplate=_hv,
                                       marker=dict(size=3.4, color=f"rgba({_rgb},0.92)")))
        try:
            _iss = df_api[df_api["Satellite Name"].str.upper().str.startswith("ISS")].iloc[0]
            fig.add_trace(go.Scatter(x=[_iss["Inclination (°)"]], y=[_iss["Altitude (km)"]], mode="markers+text", text=["ISS"],
                                     textposition="top right", textfont=dict(color="#f59e0b", size=11), name="ISS",
                                     marker=dict(symbol="diamond", size=11, color="#f59e0b", line=dict(color="#fff", width=1)), hovertemplate=_hv))
        except Exception:
            pass
        fig.add_vrect(x0=96.5, x1=99.5, fillcolor="rgba(245,158,11,0.06)", line_width=0,
                      annotation_text=T("GÜNEŞ-SENKRON", "SUN-SYNC"), annotation_position="top left",
                      annotation_font=dict(color="#f59e0b", size=10))
        fig.add_hline(y=550, line=dict(color="rgba(148,163,184,.35)", width=1, dash="dot"),
                      annotation_text="550 km", annotation_position="bottom right", annotation_font=dict(color="#8195b0", size=10))
        fig.update_layout(template="plotly_dark", height=470, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,11,20,.55)",
                          margin=dict(l=10, r=14, t=40, b=10), hovermode="closest",
                          legend=dict(orientation="h", y=1.09, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
        _ax = dict(gridcolor="rgba(34,211,238,.10)", zeroline=False, linecolor="rgba(34,211,238,.35)", ticks="outside",
                   tickcolor="rgba(34,211,238,.35)", showspikes=True, spikecolor="#22d3ee", spikethickness=1, spikedash="dot", spikemode="across")
        fig.update_xaxes(title=_xl, **_ax)
        fig.update_yaxes(title=_yl, **_ax)
        try:
            fig.update_xaxes(minor=dict(showgrid=True, gridcolor="rgba(34,211,238,.04)"))
            fig.update_yaxes(minor=dict(showgrid=True, gridcolor="rgba(34,211,238,.04)"))
        except Exception:
            pass
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    else:
        st.info(T("Şu an gösterilecek yörünge verisi yok (yedek TLE'ler çok eski olabilir).",
                  "There is no orbit data to show right now (the fallback TLEs may be too old)."))

    st.markdown(T("#### 📡 Canlı Takip Tablosu", "#### 📡 Live Tracking Table"))
    def _ops_table(df, max_h=400):
        if df is None or df.empty:
            return ""
        num = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
        th = "".join(f'<th class="{"n" if n else ""}">{html_escape(str(c))}</th>' for c, n in zip(df.columns, num))
        rows = []
        for rec in df.itertuples(index=False):
            tds = []
            for j, (v, n) in enumerate(zip(rec, num)):
                sv = html_escape(str(v))
                u = str(v).upper()
                if df.columns[j] in ("Status", "Durum"):
                    k = "ok" if ("CREWED" in u or "MÜRETTEBATLI" in u) else ("warn" if ("DECAY" in u or "ALÇAL" in u) else "nom")
                    tds.append(f'<td><span class="ops-chip {k}">{sv}</span></td>')
                elif j == 0:
                    tds.append(f'<td class="nm">{sv}</td>')
                else:
                    tds.append(f'<td class="{"n" if n else ""}">{sv}</td>')
            rows.append("<tr>" + "".join(tds) + "</tr>")
        bar = f'<div class="ops-tbl-bar"><span>{T("TAKİPTEKİ NESNELER", "TRACKED OBJECTS")} · <b>{len(df)}</b></span><span class="ops-led c" style="padding:3px 9px">{T("CANLI", "LIVE")}</span></div>'
        return _flat(f'{bar}<div class="ops-tbl-wrap" style="max-height:{max_h}px"><table class="ops-tbl"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    st.markdown(_ops_table(_disp(df_api)), unsafe_allow_html=True)

    # =====================================================================
    # EK MODÜL: KESSLER SENDROMU — ZİNCİRLEME KAZA RİSKİ VE TEMİZLİK ÖNCELİĞİ
    # Yukarıdaki hiçbir bölüme dokunmaz; yalnızca CAT (CelesTrak) verisini OKUR.
    # Değişken / widget adları kc_ / KC_ / sg_kc_ önekiyle ayrılmıştır.
    # =====================================================================
    st.markdown("<hr style='border: 1px solid rgba(252,61,33,0.30); margin: 28px 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown(T("#### 🧨 Kessler Sendromu: Zincirleme Kaza Riski ve Temizlik Önceliği",
                  "#### 🧨 Kessler Syndrome: Cascade Collision Risk & Cleanup Priority"))
    st.caption(T("Canlı CelesTrak verisi (aktif uydular + enkaz bulutları) ile yörünge kabuklarının çarpışma-zincirleme riski hesaplanır; "
                 "güncel risk bulguları ve literatür kaynakları ayrı sekmededir. Eğitim amaçlı, basitleştirilmiş bir modeldir.",
                 "Cascade risk of each orbital shell is computed from live CelesTrak data (active satellites + debris clouds); "
                 "current risk findings and literature sources are on a separate tab. This is a simplified, educational model."))

    # ---- Model sabitleri (gaz-kinetik / kabuk modeli) ----
    KC_EDGES = np.arange(300, 1550, 50)                         # 300–1500 km, 50 km'lik kabuklar
    KC_MID = (KC_EDGES[:-1] + KC_EDGES[1:]) / 2.0
    KC_VOL = 4.0 * np.pi * (RE + KC_MID) ** 2 * 50.0            # kabuk hacmi, km³
    KC_V_REL = 10.0 * 3.156e7                                   # ≈10 km/s göreli hız -> km/yıl
    KC_SIG_LL = math.pi * (2 * 1.6e-3) ** 2                     # büyük–büyük kesit alanı (r≈1,6 m), km²
    KC_SIG_LS = math.pi * (1.6e-3 + 0.15e-3) ** 2               # büyük–parça (≥10 cm sınıfı), km²
    KC_F_LL, KC_F_LS = 1500, 300                                # çarpışma başına üretilen katalog boyutlu parça
    KC_AVOID = 0.999                                            # aktif uyduların kaçınma manevrası başarısı
    KC_R_TOTAL = 1200                                           # terk edilmiş büyük nesne (≈2.000.000 kg / 1800 kg, McKnight+2021)
    KC_TAU_H = [300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 1200, 1500]
    KC_TAU_Y = [0.05, 0.2, 0.8, 2, 5, 12, 30, 60, 120, 400, 1200, 3000, 8000, 20000]   # kaba parça ömrü, yıl
    KC_TAU = np.exp(np.interp(KC_MID, KC_TAU_H, np.log(KC_TAU_Y)))

    def kc_rates(A, R, S):
        """Kabuk başına yıllık beklenen çarpışma sayıları. A: aktif uydu, R: terk edilmiş büyük nesne, S: parça."""
        k = KC_V_REL / KC_VOL
        u = 1.0 - KC_AVOID                                       # aktifler manevra yapar; yalnızca kaçamayan kısım çarpışır
        rr = 0.5 * R * R * KC_SIG_LL * k
        rs = R * S * KC_SIG_LS * k
        ar = A * R * KC_SIG_LL * k * u
        a_s = A * S * KC_SIG_LS * k * u
        aa = 0.5 * A * A * KC_SIG_LL * k * u
        return rr, rs, ar, a_s, aa

    def kc_yield(A, R, S):
        """(büyük çarpışma/yıl, üretilen yeni parça/yıl) — kabuk başına."""
        rr, rs, ar, a_s, aa = kc_rates(A, R, S)
        big = rr + ar + aa
        return big, KC_F_LL * big + KC_F_LS * (rs + a_s)

    def kc_simulate(A0, R0, S0, years, remove_per_yr, new_per_yr, strategy, dt=0.5):
        """Beklenen-değer (deterministik) kabuk simülasyonu. strategy: none | random | targeted."""
        A, R, S = A0.astype(float).copy(), R0.astype(float).copy(), S0.astype(float).copy()
        shape = R0 / max(float(R0.sum()), 1e-9)
        n = int(round(years / dt))
        tot, cum_big, cum_frag = [A.sum() + R.sum() + S.sum()], [0.0], [0.0]
        removed = np.zeros_like(R)
        for _ in range(n):
            rr, rs, ar, a_s, aa = kc_rates(A, R, S)
            big = rr + ar + aa
            Y = KC_F_LL * big + KC_F_LS * (rs + a_s)
            S = np.clip(S + dt * (Y - S / KC_TAU - rs - a_s), 0, 1e7)
            R = np.clip(R + dt * (-(2 * rr + ar + rs) - R / KC_TAU + new_per_yr * shape), 0, 1e6)
            todo = remove_per_yr * dt if strategy != "none" else 0.0
            while todo > 1e-9 and R.sum() > 1e-6:
                amt = min(1.0, todo)
                if strategy == "random":
                    take = amt * R / R.sum()
                else:
                    # Liou & Johnson: kütle × çarpışma olasılığı en yüksek hedef önce -> en çok parça üreten kabuk
                    rr2, rs2, ar2, _, _ = kc_rates(A, R, S)
                    pr = (KC_F_LL * (2 * rr2 + ar2) + KC_F_LS * rs2) / np.maximum(R, 1e-9)
                    pr[R < 1.0] = -1.0
                    b = int(np.argmax(pr))
                    if pr[b] < 0:
                        b = int(np.argmax(R))
                    take = np.zeros_like(R)
                    take[b] = min(amt, R[b])
                R -= take
                removed += take
                todo -= amt
            tot.append(A.sum() + R.sum() + S.sum())
            cum_big.append(cum_big[-1] + float(big.sum()) * dt)
            cum_frag.append(cum_frag[-1] + float(Y.sum()) * dt)
        return {"t": np.arange(n + 1) * dt, "tot": np.array(tot), "big": cum_big[-1], "frag": cum_frag[-1], "removed": removed}

    @st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
    def kc_debris_elements(fetched, _tles):
        # TLE'den ortalama irtifa / perigee / apogee / eğim (SGP4 yayılımı gerekmez); fetched yalnızca önbellek anahtarı
        rows, mu = [], 398600.4418
        for name, l1, l2 in _tles:
            try:
                s = Satrec.twoline2rv(l1, l2)
                n = s.no_kozai / 60.0                            # rad/dk -> rad/sn
                if n <= 0:
                    continue
                a = (mu / n ** 2) ** (1.0 / 3.0)
                rows.append((name, norad_id(l1), a - RE, a * (1 - s.ecco) - RE, a * (1 + s.ecco) - RE, math.degrees(s.inclo)))
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["name", "norad", "alt", "perigee", "apogee", "inc"])

    KC_CLOUDS = (("FENGYUN 1C", "Fengyun-1C · 2007"), ("COSMOS 2251", "Cosmos 2251 · 2009"),
                 ("IRIDIUM 33", "Iridium 33 · 2009"), ("COSMOS 1408", "Cosmos 1408 · 2021"))

    def kc_cloud(name):
        u = str(name).upper()
        for key, lab in KC_CLOUDS:
            if key in u:
                return lab
        return T("Diğer enkaz", "Other debris")

    def kc_band(i):
        return f"{int(KC_EDGES[i])}–{int(KC_EDGES[i + 1])} km"

    # ---- Canlı durumdan başlangıç popülasyonu ----
    kc_A0 = np.histogram(CAT["a_alt"], KC_EDGES)[0].astype(float)     # aktif uydular (CelesTrak 'active')
    kc_S0 = np.histogram(CAT["d_alt"], KC_EDGES)[0].astype(float)     # enkaz parçaları (CelesTrak debris grupları)
    kc_w = np.where(KC_MID >= 500, kc_S0, 0.0)                        # büyük terk edilmiş nesneler enkaz profilini izler (varsayım)
    if kc_w.sum() <= 0:
        kc_w = np.where((KC_MID >= 700) & (KC_MID <= 1000), 1.0, 0.0)
    kc_R0 = KC_R_TOTAL * kc_w / kc_w.sum()

    kc_big, kc_Y = kc_yield(kc_A0, kc_R0, kc_S0)
    kc_N = kc_A0 + kc_R0 + kc_S0
    kc_K = kc_Y * KC_TAU / np.maximum(kc_N, 1.0)                      # kararsızlık oranı: ömür boyu üretilen parça / mevcut nesne
    kc_i_max = int(np.argmax(kc_K))
    kc_n_crit = int(np.sum(kc_K > 1.0))
    kc_big_sum = float(kc_big.sum())

    if str(CAT.get("status_label", "")).startswith("OFFLINE"):
        st.info(T("Canlı veri alınamadı; analiz yedek verilerle çalışıyor, sonuçlar yalnızca gösterge niteliğindedir.",
                  "Live data could not be fetched; the analysis runs on fallback data and results are indicative only."))

    kc_cards = [
        (T("Kritik Kabuk (K>1)", "Critical Shells (K>1)"), f"{kc_n_crit}", "color:#fc3d21;"),
        (T("En Riskli Bant", "Riskiest Band"), kc_band(kc_i_max) if kc_K[kc_i_max] > 0 else "—", "color:#f59e0b;font-size:20px;"),
        (T("Büyük Çarpışma (model)", "Major Collision (model)"),
         (T(f"≈{1.0 / kc_big_sum:.0f} yılda 1", f"≈1 per {1.0 / kc_big_sum:.0f} yr") if kc_big_sum > 0 else "—"), "color:#f59e0b;font-size:18px;"),
        (T("CRASH Clock (26 Oca 2026)", "CRASH Clock (26 Jan 2026)"), T("3,8 gün", "3.8 days"), "color:#fc3d21;"),
    ]
    for _col, (_t, _v, _s) in zip(st.columns(4), kc_cards):
        with _col:
            st.markdown(f'<div class="nasa-glass-card"><div class="metric-title">{_t}</div><div class="metric-value" style="{_s}">{_v}</div></div>',
                        unsafe_allow_html=True)

    try:
        kc_tabs = st.tabs([T("📊 Canlı Risk Haritası", "📊 Live Risk Map"), T("🧹 Temizlik Önceliği", "🧹 Cleanup Priority"),
                           T("🎛️ Simülasyon", "🎛️ Simulation"), T("📚 Güncel Riskler ve Kaynaklar", "📚 Current Risks & Sources")])
    except AttributeError:
        kc_tabs = [st.container() for _ in range(4)]

    # ---------------------------------------------------------------- 1) Canlı risk haritası
    with kc_tabs[0]:
        kc_c1, kc_c2 = st.columns(2)
        with kc_c1:
            kc_f1 = go.Figure()
            kc_f1.add_bar(x=KC_MID, y=kc_A0, name=T("Aktif uydu", "Active sats"), marker_color="#38bdf8")
            kc_f1.add_bar(x=KC_MID, y=kc_R0, name=T("Terk edilmiş büyük nesne (tahmini)", "Derelict large objects (est.)"), marker_color="#f59e0b")
            kc_f1.add_bar(x=KC_MID, y=kc_S0, name=T("Enkaz parçası", "Debris fragments"), marker_color="#fc3d21")
            kc_f1.update_layout(barmode="stack", template="plotly_dark", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                title=T("İrtifa kabuğuna göre nesne sayısı", "Objects per altitude shell"),
                                xaxis_title=T("İrtifa (km)", "Altitude (km)"), yaxis_title=T("Nesne sayısı", "Objects"),
                                legend=dict(orientation="h", y=-0.28), margin=dict(t=50, b=10))
            st.plotly_chart(kc_f1, **stretch(st.plotly_chart))
        with kc_c2:
            kc_f2 = go.Figure(go.Bar(x=KC_MID, y=np.maximum(kc_K, 1e-4), customdata=kc_K,
                                     marker_color=["#fc3d21" if k > 1 else "#38bdf8" for k in kc_K],
                                     hovertemplate="%{x} km<br>K = %{customdata:.3f}<extra></extra>"))
            kc_f2.add_hline(y=1, line_dash="dash", line_color="#fbbf24")
            kc_f2.update_layout(template="plotly_dark", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                title=T("Kararsızlık oranı K (K>1: zincirleme eğilimi)", "Instability ratio K (K>1: cascade tendency)"),
                                xaxis_title=T("İrtifa (km)", "Altitude (km)"), yaxis_title="K", yaxis_type="log", margin=dict(t=50, b=10))
            st.plotly_chart(kc_f2, **stretch(st.plotly_chart))
        st.caption(T("K = (bir kabukta yıllık üretilen yeni parça × parçanın ortalama yörünge ömrü) ÷ kabukta şu an bulunan nesne sayısı. "
                     "K>1 ise çarpışmalarla üretilen parça, doğal olarak atmosfere düşenden hızlı artar (ESA'nın 'çarpışmalar çarpışmaları doğurur' tanımı). "
                     "Aktif uydu ve enkaz sayıları canlı CelesTrak verisidir; terk edilmiş büyük nesne sayısı McKnight+2021'den (≈1.200) tahmindir.",
                     "K = (new fragments produced per year in a shell × mean orbital lifetime of a fragment) ÷ objects currently in the shell. "
                     "If K>1, collision-made fragments grow faster than natural decay removes them (ESA's 'collisions causing collisions'). "
                     "Active-satellite and debris counts are live CelesTrak data; the derelict large-object count is an estimate (≈1,200) from McKnight+2021."))
        with st.expander(T("Model varsayımları", "Model assumptions")):
            st.markdown(T(
                "- Gaz-kinetik kabuk modeli: çarpışma/yıl = ½·N²·σ·v/V (aynı sınıf), N₁·N₂·σ·v/V (farklı sınıf); v ≈ 10 km/s.\n"
                "- Kesit alanı: büyük nesne yarıçapı ≈ 1,6 m, parça ≈ 15 cm. Çarpışma başına ≈ 1.500 (büyük–büyük) / ≈ 300 (büyük–parça) katalog boyutlu parça.\n"
                "- Aktif uydular manevra yapar: çarpışma oranı (1 − %99,9) ile çarpılır. Terk edilmiş nesneler ve parçalar manevra yapamaz.\n"
                "- Parça ömrü irtifaya göre kaba tahmindir (≈12 yıl @550 km, ≈400 yıl @800 km, ≈3.000 yıl @1000 km); Güneş döngüsüne ve alan/kütle oranına bağlıdır.\n"
                "- Yalnızca ≥10 cm sınıfı ve seçili CelesTrak grupları sayılır; katalog dışı küçük enkaz ve diğer parçalanma olayları dahil değildir (risk alt sınırdır).\n"
                "- Sonuçlar tahmin değil, ölçek ve karşılaştırma içindir.",
                "- Gas-kinetic shell model: collisions/yr = ½·N²·σ·v/V (same class), N₁·N₂·σ·v/V (different classes); v ≈ 10 km/s.\n"
                "- Cross-section: large-object radius ≈ 1.6 m, fragment ≈ 15 cm. Each collision yields ≈ 1,500 (large–large) / ≈ 300 (large–fragment) catalog-size fragments.\n"
                "- Active satellites manoeuvre: their collision rate is scaled by (1 − 99.9%). Derelicts and fragments cannot manoeuvre.\n"
                "- Fragment lifetime vs altitude is rough (≈12 yr @550 km, ≈400 yr @800 km, ≈3,000 yr @1000 km); it depends on solar cycle and area-to-mass ratio.\n"
                "- Only the ≥10 cm class and the selected CelesTrak groups are counted; uncatalogued small debris and other breakups are excluded (risk is a lower bound).\n"
                "- Results are for scale and comparison, not a forecast."))

    # ---------------------------------------------------------------- 2) Temizlik önceliği
    with kc_tabs[1]:
        kc_deb = kc_debris_elements(CAT["fetched"], CAT["debris"])
        if kc_deb.empty:
            st.info(T("Enkaz yörünge verisi yok.", "No debris orbit data."))
        else:
            kc_deb = kc_deb[(kc_deb["alt"] >= KC_EDGES[0]) & (kc_deb["alt"] < KC_EDGES[-1])].copy()
            _bi = np.clip(np.digitize(kc_deb["alt"].to_numpy(), KC_EDGES) - 1, 0, len(KC_MID) - 1)
            kc_deb["life"] = np.exp(np.interp(kc_deb["alt"].to_numpy(), KC_TAU_H, np.log(KC_TAU_Y)))
            kc_deb["expo"] = (kc_N / KC_VOL)[_bi] * KC_SIG_LS * KC_V_REL * kc_deb["life"]     # ömür boyu beklenen çarpışma
            kc_deb["p"] = (1.0 - np.exp(-kc_deb["expo"])) * 100.0
            kc_deb["cloud"] = kc_deb["name"].map(kc_cloud)

            st.markdown(T("**1) Hangi enkaz bulutu / bölge önce temizlenmeli?** (canlı CelesTrak verisi)",
                          "**1) Which debris cloud / region should be cleaned first?** (live CelesTrak data)"))
            kc_tot_expo = max(float(kc_deb["expo"].sum()), 1e-12)
            kc_grp = (kc_deb.groupby("cloud").agg(n=("name", "size"), alt=("alt", "median"), inc=("inc", "median"),
                                                  life=("life", "median"), expo=("expo", "sum")).reset_index())
            kc_grp["share"] = kc_grp["expo"] / kc_tot_expo * 100.0
            kc_grp = kc_grp.sort_values("share", ascending=False).reset_index(drop=True)

            def kc_prio(sh):
                return (T("🔴 Çok yüksek", "🔴 Very high") if sh >= 35 else T("🟠 Yüksek", "🟠 High") if sh >= 15
                        else T("🟡 Orta", "🟡 Medium") if sh >= 5 else T("⚪ Düşük", "⚪ Low"))

            st.dataframe(pd.DataFrame({
                T("Enkaz bulutu", "Debris cloud"): kc_grp["cloud"],
                T("Nesne", "Objects"): kc_grp["n"].map(_fmt_int),
                T("Medyan irtifa (km)", "Median alt. (km)"): kc_grp["alt"].round().astype(int),
                T("Eğim (°)", "Incl. (°)"): kc_grp["inc"].round(1),
                T("Tahmini ömür (yıl)", "Est. lifetime (yr)"): kc_grp["life"].map(lambda v: _fmt_int(int(round(v)))),
                T("Risk payı (%)", "Risk share (%)"): kc_grp["share"].round(1),
                T("Temizlik önceliği", "Cleanup priority"): kc_grp["share"].map(kc_prio),
            }), hide_index=True, **stretch(st.dataframe))
            st.caption(T("Risk payı = bulut içindeki nesnelerin 'yoğunluk × ömür' maruziyetlerinin toplamı (yani kalan ömürleri boyunca beklenen çarpışma). "
                         "Yüksek irtifadaki bulutlar atmosferde kendiliğinden temizlenmediği için öncelik kazanır.",
                         "Risk share = sum of each object's 'density × lifetime' exposure within the cloud (expected collisions over its remaining life). "
                         "High-altitude clouds rank higher because the atmosphere will not clear them on its own."))

            st.markdown(T("**2) Çarpışma ihtimali en yüksek 10 enkaz parçası** (canlı)", "**2) Top 10 debris pieces by collision probability** (live)"))
            kc_top = kc_deb.sort_values("p", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(pd.DataFrame({
                T("Enkaz", "Debris"): kc_top["name"], "NORAD": kc_top["norad"], T("Bulut", "Cloud"): kc_top["cloud"],
                T("Ort. irtifa (km)", "Mean alt. (km)"): kc_top["alt"].round().astype(int),
                T("Perigee–Apogee (km)", "Perigee–Apogee (km)"): [f"{int(p)}–{int(a)}" for p, a in zip(kc_top["perigee"], kc_top["apogee"])],
                T("Eğim (°)", "Incl. (°)"): kc_top["inc"].round(1),
                T("Tahmini ömür (yıl)", "Est. lifetime (yr)"): kc_top["life"].map(lambda v: _fmt_int(int(round(v)))),
                T("Ömür boyu çarpışma ihtimali (%)", "Lifetime collision prob. (%)"): kc_top["p"].round(1),
            }), hide_index=True, **stretch(st.dataframe))
            st.caption(T("Bunlar ≈10 cm sınıfı parçalardır ve tek tek kaldırılmaları pratik değildir; tablo hangi bölgenin/bulutun tehlikeli olduğunu gösterir. "
                         "Kütle bilinmediği için 'kütle × çarpışma olasılığı' ölçütünde kütle eşit alınmıştır. Gerçek temizlik hedefleri için aşağıdaki literatür tablosuna bakın.",
                         "These are ≈10 cm-class fragments and removing them one by one is impractical; the table shows which region/cloud is dangerous. "
                         "Mass is unknown, so the 'mass × collision probability' criterion treats mass as equal. For real removal targets see the literature table below."))

        st.markdown(T("**3) Literatürdeki öncelikli kaldırma hedefleri** (McKnight ve ark., 'en endişe verici 50 nesne' — 2025 güncellemesi, ilk 10)",
                      "**3) Priority removal targets in the literature** (McKnight et al., 'most-concerning 50 objects' — 2025 update, top 10)"))
        kc_ctry = (["Rusya", "ESA", "Japonya", "Çin", "SSCB", "SSCB", "Rusya", "Rusya", "SSCB", "Çin"] if st.session_state.lang == "tr"
                   else ["Russia", "ESA", "Japan", "China", "USSR", "USSR", "Russia", "Russia", "USSR", "China"])
        st.dataframe(pd.DataFrame({
            T("Sıra", "Rank"): list(range(1, 11)),
            T("Nesne", "Object"): ["SL-16 R/B", "Envisat", "H-II R/B", "CZ-2C R/B", "SL-8 R/B", "SL-16 R/B", "Kosmos 2237", "Kosmos 2334", "SL-16 R/B", "CZ-2D R/B"],
            T("Tür", "Type"): [T("Roket gövdesi", "Rocket body"), T("Uydu (işlevsiz)", "Satellite (defunct)"), T("Roket gövdesi", "Rocket body"),
                               T("Roket gövdesi", "Rocket body"), T("Roket gövdesi", "Rocket body"), T("Roket gövdesi", "Rocket body"),
                               T("Uydu (işlevsiz)", "Satellite (defunct)"), T("Uydu (işlevsiz)", "Satellite (defunct)"),
                               T("Roket gövdesi", "Rocket body"), T("Roket gövdesi", "Rocket body")],
            T("Ülke / kurum", "Country / agency"): kc_ctry,
            T("Fırlatma", "Launch"): ["2004", "2002", "1996", "—", "1985", "1988", "1993", "1996", "1988", "2019"],
        }), hide_index=True, **stretch(st.dataframe))
        st.caption(T("SL-16 (Zenit 2. kademe): ≈9.000 kg, ≈11 m × 4 m (LeoLabs). En olası çarpışma bölgesi 775–850 km. "
                     "Seçim ölçütü Liou & Johnson'a göre en yüksek kütle × çarpışma olasılığıdır. Ayrıntı ve bağlantılar 'Güncel Riskler ve Kaynaklar' sekmesinde.",
                     "SL-16 (Zenit 2nd stage): ≈9,000 kg, ≈11 m × 4 m (LeoLabs). The most likely collision region is 775–850 km. "
                     "Per Liou & Johnson the selection criterion is the highest mass × collision probability. Details and links are on the 'Current Risks & Sources' tab."))

    # ---------------------------------------------------------------- 3) Simülasyon
    with kc_tabs[2]:
        st.markdown(T("Seçtiğin temizlik hızıyla önümüzdeki yıllarda nesne sayısının nasıl değişeceğini karşılaştır.",
                      "Compare how the object count evolves over the coming years at the cleanup rate you choose."))
        kc_s1, kc_s2, kc_s3 = st.columns(3)
        kc_years = kc_s1.slider(T("Simülasyon süresi (yıl)", "Simulation horizon (years)"), 10, 100, 50, 5, key="sg_kc_years")
        kc_rem = kc_s2.slider(T("Yılda temizlenen büyük nesne", "Large objects removed per year"), 0, 20, 5, key="sg_kc_rem")
        kc_new = kc_s3.slider(T("Yılda yeni terk edilen büyük nesne", "New large objects abandoned per year"), 0, 40, 5, key="sg_kc_new")

        kc_runs = {
            "none": kc_simulate(kc_A0, kc_R0, kc_S0, kc_years, 0, kc_new, "none"),
            "random": kc_simulate(kc_A0, kc_R0, kc_S0, kc_years, kc_rem, kc_new, "random"),
            "targeted": kc_simulate(kc_A0, kc_R0, kc_S0, kc_years, kc_rem, kc_new, "targeted"),
        }
        kc_names = {"none": T("Temizlik yok", "No cleanup"), "random": T("Rastgele temizlik", "Random cleanup"),
                    "targeted": T("Hedefli temizlik (en riskli bant)", "Targeted cleanup (riskiest band)")}
        kc_colors = {"none": "#fc3d21", "random": "#f59e0b", "targeted": "#10b981"}

        kc_g1, kc_g2 = st.columns([3, 2])
        with kc_g1:
            kc_f3 = go.Figure()
            for _k, _r in kc_runs.items():
                kc_f3.add_trace(go.Scatter(x=_r["t"], y=_r["tot"], mode="lines", name=kc_names[_k], line=dict(color=kc_colors[_k], width=3)))
            kc_f3.update_layout(template="plotly_dark", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                title=T("Toplam nesne sayısı (aktif + terk edilmiş + parça)", "Total objects (active + derelict + fragments)"),
                                xaxis_title=T("Yıl", "Year"), yaxis_title=T("Nesne sayısı", "Objects"),
                                legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10))
            st.plotly_chart(kc_f3, **stretch(st.plotly_chart))
        with kc_g2:
            kc_rm = kc_runs["targeted"]["removed"]
            kc_f4 = go.Figure(go.Bar(x=KC_MID, y=kc_rm, marker_color="#10b981"))
            kc_f4.update_layout(template="plotly_dark", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                title=T("Hedefli senaryoda nereden temizlendi?", "Where did the targeted scenario clean?"),
                                xaxis_title=T("İrtifa (km)", "Altitude (km)"), yaxis_title=T("Kaldırılan nesne", "Objects removed"), margin=dict(t=50, b=10))
            st.plotly_chart(kc_f4, **stretch(st.plotly_chart))

        def kc_state(chg):
            return (T("🔴 Kontrolsüz artış", "🔴 Runaway growth") if chg > 25 else T("🟠 Artış sürüyor", "🟠 Still growing") if chg > 5
                    else T("🟡 Yaklaşık sabit", "🟡 Roughly flat") if chg > -5 else T("🟢 Azalıyor", "🟢 Declining"))

        kc_rows = []
        for _k, _r in kc_runs.items():
            _chg = (_r["tot"][-1] / max(_r["tot"][0], 1.0) - 1.0) * 100.0
            kc_rows.append({T("Senaryo", "Scenario"): kc_names[_k], T("Sonda nesne sayısı", "Final objects"): _fmt_int(int(round(_r["tot"][-1]))),
                            T("Değişim (%)", "Change (%)"): round(_chg, 1), T("Üretilen yeni parça", "New fragments made"): _fmt_int(int(round(_r["frag"]))),
                            T("Büyük çarpışma (beklenen)", "Major collisions (expected)"): round(_r["big"], 2), T("Durum", "Status"): kc_state(_chg)})
        st.dataframe(pd.DataFrame(kc_rows), hide_index=True, **stretch(st.dataframe))

        kc_share = kc_rm / max(float(kc_rm.sum()), 1e-9) * 100.0
        if kc_rem > 0 and kc_rm.sum() > 0:
            _top = np.argsort(kc_share)[::-1][:2]
            _txt = ", ".join(f"{kc_band(int(i))} (%{kc_share[i]:.0f})" for i in _top if kc_share[i] > 0.5)
            st.success(T(f"Model önerisi: temizlik kapasiteni önce şu bantlara yönelt → {_txt}. Büyük nesneleri bu bantlardan almak en çok yeni parçayı önler.",
                         f"Model suggestion: point your cleanup capacity at these bands first → {_txt}. Removing large objects there prevents the most new fragments."))
        kc_base_chg = (kc_runs["none"]["tot"][-1] / max(kc_runs["none"]["tot"][0], 1.0) - 1.0) * 100.0
        st.caption(T(f"Literatür karşılaştırması: NASA (Liou & Johnson) LEO'yu sabit tutmak için yılda ≈5 büyük nesne kaldırılmasını gerektiğini bulmuştur; "
                     f"bu model temizlik yokken {kc_years} yılda %{kc_base_chg:+.0f} değişim gösteriyor. Eğitim amaçlı beklenen-değer modelidir.",
                     f"Literature check: NASA (Liou & Johnson) found that ≈5 large objects/year must be removed to hold LEO constant; "
                     f"this model shows {kc_base_chg:+.0f}% change over {kc_years} years without cleanup. Educational expected-value model."))

    # ---------------------------------------------------------------- 4) Güncel riskler ve kaynaklar
    with kc_tabs[3]:
        st.markdown(T("**Şu anda zincirleme kazaya yol açabilecek riskler** (web araştırması, 14 Kasım 2026 itibarıyla)",
                      "**Risks that could trigger a chain of collisions right now** (web research, as of 14 November 2026)"))
        kc_tr = st.session_state.lang == "tr"
        kc_src = [
            ("ESA Space Environment Report 2026", "https://www.esa.int/articles/esa-space-environment-report-2026"),
            ("arXiv:2512.09643 (Thiele ve ark.)", "https://arxiv.org/abs/2512.09643"),
            ("The Weather Network — CRASH Clock", "https://www.theweathernetwork.com/en/news/science/space/crash-clock-tracks-collision-risks-in-low-earth-orbit"),
            ("KeepTrack — USA 32 breakup notice", "https://keeptrack.space/ktoc-notice/kt-20260913-19460-ann-usa-32-breakup-announced-by-space-track"),
            ("McKnight ve ark. 2021, Acta Astronautica 181", "https://ui.adsabs.harvard.edu/abs/2021AcAau.181..282M/abstract"),
            ("LeoLabs — SL-16 / Cluster 850", "https://leolabs.space/?p=3822"),
            ("SpaceNews — 775–850 km bölgesi", "https://spacenews.com/?p=123624"),
            ("Ars Technica alıntısı (Slashdot) — 2025 top-50", "https://tech.slashdot.org/story/25/10/06/0148246"),
            ("NASA NTRS — Liou, Active Debris Removal", "https://ntrs.nasa.gov/citations/20120013266"),
            ("Astroscale — ADRAS-J2", "https://astroscale.com/en/news/astroscale-japan-selects-isar-aerospace-to-launch-adras-j2"),
            ("CelesTrak (canlı TLE)" if kc_tr else "CelesTrak (live TLE)", "https://celestrak.org/"),
        ]
        kc_L = lambda i: f"[{kc_src[i][0]}]({kc_src[i][1]})"
        kc_rows_md = [
            (T("Uzay ortamı sağlık endeksi", "Space environment health index"),
             T("ESA endeksi bir yılda ≈4'ten ≈50'ye çıktı (sürdürülebilir hedefin 50 katı baskı). Nesne ve çarpışma projeksiyonları 'kontrolsüz' artıyor; ufuk 200'den 100 yıla indirildi.",
               "ESA's index jumped from ≈4 to ≈50 in one year (50× the sustainable target). Object and collision projections are 'runaway'; the horizon was cut from 200 to 100 years."),
             "14.09.2026", kc_L(0)),
            (T("Fırlatma temposu", "Launch pace"),
             T("2025'te 300+ fırlatma ve 4.000+ yeni yük; günde ≈10 yeni yük. Yeterli sayıda uydu, ömrü bitince kalabalık yörüngelerden ayrılmıyor.",
               "300+ launches and 4,000+ new payloads in 2025; ≈10 new payloads per day. Not enough satellites leave congested orbits at end of life."),
             "14.09.2026", kc_L(0)),
            (T("ADR zorunluluğu", "ADR is required"),
             T("ESA: önleme (trafik yönetimi, ömür sonu bertaraf, Zero Debris) şart ama tek başına yetmez; çarpışmaların çarpışma doğurmasını (Kessler) durdurmak için aktif enkaz kaldırma gerekli.",
               "ESA: prevention (traffic management, end-of-life disposal, Zero Debris) is essential but no longer enough; active debris removal is needed to stop collisions causing collisions (Kessler)."),
             "14.09.2026", kc_L(0)),
            (T("CRASH Clock", "CRASH Clock"),
             T("Manevra yapılamazsa yıkıcı çarpışmaya kalan süre: 2018'de 164 gün → 5,5 gün (Haz 2025 verisi). 26 Oca 2026 güncellemesi: 3,8 gün. (İlk ön baskıda 2,8 gün raporlanmıştı; metrik revize edildi.)",
               "Time to a catastrophic collision if manoeuvres stop: 164 days in 2018 → 5.5 days (June 2025 data). 26 Jan 2026 update: 3.8 days. (The first preprint reported 2.8 days; the metric was revised.)"),
             "2025–2026", f"{kc_L(1)} · {kc_L(2)}"),
            (T("USA 32 parçalanması", "USA 32 breakup"),
             T("1988'de fırlatılan ABD keşif uydusu (NORAD 19460) ≈775 km'de, ≈85° eğimde parçalandı; neden bilinmiyor. Bölge, en riskli 775–850 km bandının hemen içinde.",
               "A US reconnaissance satellite launched in 1988 (NORAD 19460) broke up at ≈775 km, ≈85° inclination; cause unknown. The region lies right in the riskiest 775–850 km band."),
             "13.09.2026", kc_L(3)),
            (T("Büyük terk edilmiş nesneler", "Massive derelict objects"),
             T("LEO'da >2.000.000 kg terk edilmiş kütle (ort. >1.800 kg); ≈850 km'de 36 nesnelik küme ort. >6.000 kg. SL-16 gövdesi ≈9.000 kg, 11 m × 4 m.",
               "Over 2,000,000 kg of derelict mass in LEO (avg >1,800 kg); a cluster of 36 objects near 850 km averages >6,000 kg. An SL-16 body is ≈9,000 kg, 11 m × 4 m."),
             "2021", f"{kc_L(4)} · {kc_L(5)}"),
            (T("En olası çarpışma bölgesi", "Most likely collision region"),
             T("Rus roket gövdeleri ile Çin/ABD parçaları ve işlevsiz yükler arasındaki çarpışmalar en çok 775–850 km'de bekleniyor.",
               "Collisions between Russian rocket bodies and Chinese/US fragments and defunct payloads are most likely between 775 and 850 km."),
             "—", kc_L(6)),
            (T("En tehlikeli 50 nesne (2025)", "Most-concerning 50 objects (2025)"),
             T("İlk 50'nin %88'i roket gövdesi, %76'sı geçen yüzyılda bırakılmış; 1 Oca 2024'ten beri 25 yıldan uzun kalacak 26 roket gövdesi daha terk edildi.",
               "88% of the top 50 are rocket bodies and 76% were left last century; since 1 Jan 2024 another 26 rocket bodies were abandoned that will stay >25 years."),
             "10.2025", kc_L(7)),
            (T("Gereken temizlik hızı", "Required removal rate"),
             T("NASA: LEO'yu 200 yıl sabit tutmak için yılda ≈5 büyük (1–8+ ton) nesne kaldırmak gerekir; hedefler en yüksek kütle × çarpışma olasılığına göre seçilir.",
               "NASA: holding LEO constant for 200 years needs ≈5 large (1–8+ ton) objects removed per year, chosen by highest mass × collision probability."),
             "2009–2012", kc_L(8)),
            (T("Temizlik teknolojisi", "Removal technology"),
             T("ADRAS-J2 (Astroscale/JAXA), ≈3 ton, 11 m × 4 m bir üst kademeyi yakalayıp indirmeyi hedefliyor; fırlatma en erken 2027. Henüz operasyonel ticari kaldırma hizmeti yok.",
               "ADRAS-J2 (Astroscale/JAXA) aims to capture and deorbit a ≈3-ton, 11 m × 4 m upper stage; launch NET 2027. No operational commercial removal service exists yet."),
             "01.09.2026", kc_L(9)),
        ]
        st.markdown("\n".join(
            [f"| {T('Risk', 'Risk')} | {T('Bulgu', 'Finding')} | {T('Tarih', 'Date')} | {T('Kaynak', 'Source')} |", "|---|---|---|---|"]
            + [f"| **{a}** | {b} | {c_} | {d} |" for a, b, c_, d in kc_rows_md]))
        st.caption(T(f"Canlı nesne sayıları ve yoğunluk hesabı: {kc_L(10)} (son güncelleme: {CAT['fetched']}). "
                     "Haber sitelerinden alınan satırlar ikincil kaynaktır; kritik kararlar için birincil makale/rapora bakın.",
                     f"Live object counts and density: {kc_L(10)} (last update: {CAT['fetched']}). "
                     "Rows taken from news sites are secondary sources; check the primary paper/report for critical decisions."))

# =====================================================================
# 2. MODÜL: 3D RADAR
# =====================================================================
elif st.session_state.active_page == PAGES[1]:
    st.markdown(T("### 🌍 SPACE GRAVE [SP4C3 GR4V3] Profesyonel 3D Radar Merkezi", "### 🌍 SPACE GRAVE [SP4C3 GR4V3] Professional 3D Radar Hub"))
    st.caption(T("Her nesne gerçek TLE/SATCAT verisinden alınır; konumlar yörünge elemanlarından (J2 etkili Kepler) canlı hesaplanır. Türü, fırlatma tarihini ve zaman hızını soldaki panelden / alttaki çubuktan ayarla; ⏸ ile anlık durdurabilirsin.",
                 "Every object comes from real TLE/SATCAT data; positions are computed live from orbital elements (J2-perturbed Kepler). Pick type, launch date and time speed from the left panel / bottom bar; ⏸ pauses instantly."))

    import re

    # --- Ek veri kaynakları (CelesTrak) + SATCAT (tür / durum / fırlatma tarihi) ---
    # Roket gövdesi, parça ve eski uydular için ad aramaları; hepsi diske önbelleklenir (2 sa).
    TRACKER_QUERIES = {
        "rb": {"NAME": "R/B"}, "deb": {"NAME": "DEB"}, "cosmos": {"NAME": "COSMOS"},
        "noaa": {"NAME": "NOAA"}, "fengyun": {"NAME": "FENGYUN"}, "dmsp": {"NAME": "DMSP"}, "geo": {"GROUP": "geo"},
    }
    SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
    MU_KM = 398600.4418
    J2 = 1.08262668e-3

    def _fetch_tle_query(key, params):
        path = os.path.join(CACHE_DIR, f"q_{key}.tle")
        has_cache = os.path.exists(path)
        if has_cache and time.time() - os.path.getmtime(path) < CACHE_TTL_S:
            with open(path, encoding="utf-8") as f:
                return parse_tles(f.read()), "cache"
        try:
            r = requests.get("https://celestrak.org/NORAD/elements/gp.php",
                             params={**params, "FORMAT": "tle"}, timeout=(5, 60), headers=HEADERS)
            r.raise_for_status()
            tles = parse_tles(r.text)
            if not tles:
                raise ValueError("boş yanıt")
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            return tles, "live"
        except Exception:
            if has_cache:
                with open(path, encoding="utf-8") as f:
                    return parse_tles(f.read()), "stale"
            return [], "fail"

    def _load_satcat():
        """NORAD -> (tür, işletim durumu, fırlatma tarihi). Alınamazsa {} (ad tabanlı sınıflamaya düşülür)."""
        path = os.path.join(CACHE_DIR, "satcat.csv")
        fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600
        if not fresh:
            try:
                r = requests.get(SATCAT_URL, timeout=(5, 90), headers=HEADERS)
                r.raise_for_status()
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(r.content)
            except Exception:
                if not os.path.exists(path):
                    return {}
        try:
            want = {"NORAD_CAT_ID", "OBJECT_TYPE", "OPS_STATUS_CODE", "LAUNCH_DATE", "DECAY_DATE"}
            df = pd.read_csv(path, usecols=lambda c: c in want, dtype=str)
            if not {"NORAD_CAT_ID", "OBJECT_TYPE"} <= set(df.columns):
                return {}
            for c in want - set(df.columns):
                df[c] = None
            df = df[df["DECAY_DATE"].isna()]
            return {str(n).strip(): (t, o, l) for n, t, o, l in
                    zip(df["NORAD_CAT_ID"], df["OBJECT_TYPE"], df["OPS_STATUS_CODE"], df["LAUNCH_DATE"])}
        except Exception:
            return {}

    def _kind_of(name, in_active, meta):
        """0 çalışan uydu · 1 ölü uydu · 2 roket gövdesi · 3 parça (enkaz) · 4 istasyon"""
        up = name.upper()
        if meta:
            typ, ops = meta[0], (meta[1] or "")
            if typ == "DEB":
                return 3
            if typ == "R/B":
                return 2
            if typ == "PAY":
                k = 0 if (in_active or ops in ("+", "P", "B", "S", "X")) else 1
                return 4 if k == 0 and up.startswith(CREWED_PREFIXES) else k
            if typ == "UNK":
                return 3
        if re.search(r"\bDEB\b", up):
            return 3
        if "R/B" in up:
            return 2
        if in_active:
            return 4 if up.startswith(CREWED_PREFIXES) else 0
        return 1

    @st.cache_data(ttl=600, show_spinner=False)
    def tracker_payload(fetched_key):
        cat = CAT
        active_ids = {norad_id(t[1]) for t in cat["groups"]["active"]}
        with ThreadPoolExecutor(max_workers=len(TRACKER_QUERIES) + 1) as ex:
            futs = [ex.submit(_fetch_tle_query, k, p) for k, p in TRACKER_QUERIES.items()]
            f_cat = ex.submit(_load_satcat)
            extra = [f.result()[0] for f in futs]
            satcat = f_cat.result()

        uniq = {}
        for lst in [cat["groups"]["active"], cat["groups"]["visual"], cat["debris"]] + extra:
            for t in lst:
                uniq.setdefault(norad_id(t[1]), t)

        nids, sats, tles = [], [], []
        for nid, t in uniq.items():
            try:
                sats.append(Satrec.twoline2rv(t[1], t[2]))
                nids.append(nid)
                tles.append(t)
            except Exception:
                continue

        now = utc_now()
        jd0, fr0 = to_jd(now)
        out = {"t0": int(now.timestamp() * 1000), "omega": 7.2921159e-5, "sun": _sun_unit_three(jd0 + fr0),
               "gmst0": math.radians((280.46061837 + 360.98564736629 * (jd0 + fr0 - 2451545.0)) % 360),
               "n": [], "id": [], "k": [], "age": [], "el": []}
        if not sats:
            return out

        e, r, _ = SatrecArray(sats).sgp4(np.array([jd0]), np.array([fr0]))
        rn = np.linalg.norm(r[:, 0, :], axis=1)
        no = np.array([s.no_kozai for s in sats]) / 60.0          # rad/s
        ecc = np.array([s.ecco for s in sats])
        inc = np.array([s.inclo for s in sats])
        raan = np.array([s.nodeo for s in sats])
        argp = np.array([s.argpo for s in sats])
        mo = np.array([s.mo for s in sats])
        ep = np.array([s.jdsatepoch + s.jdsatepochF for s in sats])
        ok = (e[:, 0] == 0) & (rn > RE + 120) & (rn < 60000) & (no > 0) & (ecc < 0.95)

        with np.errstate(all="ignore"):
            a_km = (MU_KM / no ** 2) ** (1.0 / 3.0)
            p_km = a_km * (1 - ecc ** 2)
            kk = 1.5 * J2 * (RE / p_km) ** 2
            ci = np.cos(inc)
            d_om = -kk * no * ci
            d_w = 0.5 * kk * no * (5 * ci ** 2 - 1)
            d_m = no * (1 + 0.5 * kk * np.sqrt(1 - ecc ** 2) * (3 * ci ** 2 - 1))
            dt = ((jd0 + fr0) - ep) * 86400.0                      # elemanların epoğundan şimdiye (sn)
            m_now = np.mod(mo + d_m * dt, 2 * math.pi)
            om_now = np.mod(raan + d_om * dt, 2 * math.pi)
            w_now = np.mod(argp + d_w * dt, 2 * math.pi)
            a_re = a_km / RE
        ok &= np.isfinite(a_re) & np.isfinite(m_now) & np.isfinite(om_now) & np.isfinite(w_now)

        today = now.date()
        names, ids, kinds, ages = [], [], [], []
        for i in np.flatnonzero(ok):
            name, l1, _ = tles[i]
            nid = nids[i]
            meta = satcat.get(nid)
            kinds.append(_kind_of(name, nid in active_ids, meta))
            age = -1
            try:
                if meta and meta[2]:
                    age = (today - datetime.strptime(meta[2][:10], "%Y-%m-%d").date()).days
                else:   # SATCAT yoksa uluslararası tanımlayıcıdan yaklaşık (yıl düzeyinde) tarih
                    yy = int(l1[9:11])
                    age = (today - datetime(2000 + yy if yy < 57 else 1900 + yy, 7, 1).date()).days
            except Exception:
                age = -1
            names.append(name.strip())
            ids.append(nid)
            ages.append(age)

        sel = np.flatnonzero(ok)
        el = np.stack([a_re[sel], ecc[sel], inc[sel], om_now[sel], w_now[sel], m_now[sel]], axis=1)
        el = np.round(el, 5)
        el[:, 1] = np.round(ecc[sel], 6)
        out.update({"n": names, "id": ids, "k": kinds, "age": ages, "el": el.ravel().tolist()})
        return out

    with st.spinner(T("Yörünge verisi hazırlanıyor (ilk açılışta indirme biraz sürebilir)...",
                      "Preparing orbit data (first load may take a moment)...")):
        tracker = tracker_payload(CAT["fetched"])
    flags = sorted(st.session_state.conj_meta["names"]) if st.session_state.conj_meta else []

    _L = {
        "loc": "tr-TR" if st.session_state.lang == "tr" else "en-US",
        "kicker": T("UZAY ÇÖPÜ TAKİPÇİSİ · CANLI GÖRÜNTÜ", "SPACE DEBRIS TRACKER · LIVE SNAPSHOT"),
        "title": T("Yörüngede Bıraktığımız Her Şey", "Everything We’ve Left in Orbit"),
        "desc": T("Her nokta; kamuya açık uzay kataloğundan, kendi yörüngesinde hareket eden ölü bir uydu, kullanılmış bir roket gövdesi ya da parçadır. Hâlâ kullanımda olanları görmek için çalışan uyduları aç.",
                  "Each dot is a dead satellite, spent rocket stage or fragment from the public space catalogue, moving on its own orbit. Turn on working satellites to see what’s still in use."),
        "big": T("parça çöp şu an Dünya çevresinde dönüyor", "pieces of junk circling Earth right now"),
        "types": T("Nesne türü", "Object type"), "launched": T("Fırlatılma", "Launched"),
        "sat": T("Çalışan uydular", "Working satellites"), "dead": T("Ölü uydular", "Dead satellites"),
        "rb": T("Roket gövdeleri", "Rocket bodies"), "deb": T("Parçalar", "Fragments"),
        "chips": {"any": T("Her zaman", "Any time"), "d30": T("Son 30 gün", "Last 30 days"),
                  "y1": T("Son 12 ay", "Last 12 months"), "y5": T("Son 5 yıl", "Last 5 years"),
                  "y10": T("Son 10 yıl", "Last 10 years"), "old": T("10 yıldan eski", "Older than 10 years")},
        "help": T("Sürükle: döndür · Tekerlek: yakınlaştır · Nesneye gel: yörüngesini gör · Tıkla: sabitle · Boşluk: duraklat / devam",
                  "Drag: rotate · Wheel: zoom · Hover: see orbit · Click: pin · Space: pause / resume"),
        "speed": T("HIZ", "SPEED"), "live": T("Canlıya dön", "Back to live"),
        "pause": T("Duraklat", "Pause"), "play": T("Devam et", "Resume"), "paused": T("DURAKLATILDI", "PAUSED"),
        "livebadge": T("CANLI", "LIVE"), "sim": T("SİMÜLASYON", "SIMULATION"), "u": [T("sn", "s"), T("dk", "min"), T("sa", "h")],
        "hud": T("🎯 CANLI YÖRÜNGE TELEMETRİSİ", "🎯 LIVE ORBITAL TELEMETRY"),
        "target": T("Hedef", "Target"), "norad": T("NORAD No", "NORAD ID"), "cls": T("Sınıf", "Class"),
        "alt": T("İrtifa", "Altitude"), "vel": T("Hız", "Velocity"), "inc": T("Eğim", "Inclination"),
        "per": T("Periyot", "Period"), "thr": T("Yakınlaşma takibi", "Conjunction watch"),
        "hover": T("BİR NESNEYE GELİN", "HOVER OVER OBJECT"),
        "cSat": T("UYDU", "SATELLITE"), "cDead": T("ÖLÜ UYDU", "DEAD SATELLITE"), "cRb": T("ROKET GÖVDESİ", "ROCKET BODY"),
        "cDeb": T("ENKAZ PARÇASI", "DEBRIS FRAGMENT"), "cStn": T("MÜRETTEBATLI İSTASYON", "CREWED STATION"),
        "flag": T("İŞARETLİ", "FLAGGED"), "noflag": T("işaretli değil", "none flagged"),
        "few": T("Veri az: CelesTrak'a ulaşılamadı, yalnızca yedek/önbellek verisi gösteriliyor. İnternet bağlantını kontrol edip sayfayı yenile.",
                 "Little data: CelesTrak was unreachable, showing only fallback/cached data. Check your connection and reload."),
        "colors": [[0.22, 0.74, 0.97], [0.65, 0.55, 0.98], [1.0, 0.70, 0.28], [0.98, 0.45, 0.45], [0.20, 0.83, 0.60]],
    }
    TRACKER_HTML = r"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{--cyan:#38bdf8;--red:#fc3d21;--green:#10b981;--mute:#94a3b8;--line:rgba(56,189,248,.16);--card:rgba(6,12,30,.80)}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#040817;overflow:hidden;font-family:'Inter','Segoe UI',-apple-system,sans-serif;color:#e2e8f0}
#app{display:flex;height:100vh;width:100%}
#panel{width:384px;flex:0 0 384px;overflow-y:auto;padding:26px 22px 26px 22px;background:linear-gradient(180deg,rgba(10,18,40,.98) 0%,rgba(4,8,23,.99) 100%);border-right:1px solid var(--line);scrollbar-width:thin;scrollbar-color:#1e3a5f #040817}
.kick{display:inline-flex;align-items:center;gap:9px;padding:7px 14px;border-radius:999px;border:1px solid var(--line);background:rgba(8,16,38,.65);font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#bae6fd;line-height:1.3}
.kick::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green);flex:0 0 7px;animation:pulse 2.2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(16,185,129,.55)}70%{box-shadow:0 0 0 9px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}
h1{font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:38px;line-height:1.1;margin:20px 0 14px;letter-spacing:-.02em;color:#fff}
.desc{color:#cbd5e1;font-size:15px;line-height:1.7;margin:0 0 22px}
.big{display:flex;align-items:center;gap:16px;margin-bottom:26px;padding:16px 18px;border-radius:18px;background:linear-gradient(160deg,rgba(14,24,48,.84),rgba(6,11,26,.92));border:1px solid var(--line)}
.big b{font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:48px;letter-spacing:-.02em;color:var(--cyan);font-variant-numeric:tabular-nums;line-height:1;text-shadow:0 0 24px rgba(56,189,248,.35)}
.big span{color:var(--mute);font-size:13.5px;line-height:1.4;max-width:150px}
.sec{font-size:12px;font-weight:700;letter-spacing:.22em;color:var(--cyan);margin:4px 0 12px;text-transform:uppercase}
.row{display:flex;align-items:center;width:100%;gap:14px;padding:11px 12px;margin:0 0 6px;background:rgba(8,16,38,.45);border:1px solid transparent;border-radius:14px;color:#fff;cursor:pointer;font:inherit;text-align:left;transition:border-color .2s,background .2s}
.row:hover{border-color:rgba(56,189,248,.45);background:rgba(14,24,48,.7)}
.row .dot{width:16px;height:16px;border-radius:50%;border:2px solid var(--c);background:var(--c);box-shadow:0 0 10px var(--c);flex:0 0 16px;transition:all .15s}
.row .nm{flex:1;font-size:16px;font-weight:500}
.row .ct{font-family:'Space Grotesk','Inter',sans-serif;color:var(--mute);font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
.row.off{color:#64748b}
.row.off .dot{background:transparent;box-shadow:none;border-color:#475569}
.row.off .ct{color:#475569}
.chips{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:22px}
.chip{padding:9px 16px;border-radius:999px;border:1px solid var(--line);background:rgba(56,189,248,.06);color:#e2e8f0;font:inherit;font-size:13.5px;font-weight:500;cursor:pointer;transition:all .2s}
.chip:hover{border-color:var(--cyan);background:rgba(56,189,248,.14)}
.chip.on{background:rgba(56,189,248,.22);border-color:var(--cyan);color:#fff;font-weight:600}
.help{color:#64748b;font-size:12px;line-height:1.7;margin-top:8px}
.warn{display:none;margin:0 0 16px;padding:10px 12px;border-radius:12px;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.4);color:#fcd34d;font-size:12.5px;line-height:1.5}
#stage{flex:1;position:relative;min-width:0;background:radial-gradient(ellipse at 60% 45%,#0b1329 0%,#040817 72%)}
#cv{position:absolute;inset:0;width:100%;height:100%;display:block;outline:none}
#clock{position:absolute;left:18px;top:16px;font-size:12.5px;font-weight:600;letter-spacing:.04em;color:#e2e8f0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px 13px;backdrop-filter:blur(10px);z-index:5;font-variant-numeric:tabular-nums}
#clock i{font-style:normal;margin-left:10px;padding:3px 9px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.14em;background:rgba(16,185,129,.16);color:var(--green)}
#clock i.sim{background:rgba(56,189,248,.16);color:#7dd3fc}
#clock i.pz{background:rgba(251,191,36,.16);color:#fbbf24}
#hud{position:absolute;right:16px;top:16px;width:300px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 18px;z-index:5;backdrop-filter:blur(12px);box-shadow:0 12px 40px rgba(0,0,0,.5)}
#hud h4{margin:0 0 10px;font-size:11px;letter-spacing:.2em;color:var(--cyan);font-weight:700;text-transform:uppercase;border-bottom:1px solid var(--line);padding-bottom:9px}
#hud p{margin:6px 0;font-size:12.5px;color:var(--mute);display:flex;justify-content:space-between;gap:10px}
#hud p span{color:#f8fafc;font-weight:600;font-size:12.5px;text-align:right;max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#bar{position:absolute;left:50%;transform:translateX(-50%);bottom:16px;display:flex;align-items:center;gap:14px;padding:12px 16px;background:var(--card);border:1px solid var(--line);border-radius:20px;z-index:6;backdrop-filter:blur(12px);box-shadow:0 12px 40px rgba(0,0,0,.55);max-width:calc(100% - 24px)}
#bar button{font:inherit;cursor:pointer;color:#e2e8f0}
#pp{display:flex;align-items:center;gap:8px;height:42px;padding:0 22px;border-radius:999px;border:1px solid rgba(252,61,33,.9);background:linear-gradient(135deg,#fc3d21 0%,#c42a12 100%);color:#fff!important;font-weight:700;font-size:14px;white-space:nowrap;box-shadow:0 8px 26px rgba(252,61,33,.3);transition:transform .2s,box-shadow .2s}
#pp:hover{transform:translateY(-1px);box-shadow:0 12px 32px rgba(252,61,33,.45)}
#pp.paused{border-color:rgba(16,185,129,.9);background:linear-gradient(135deg,#10b981 0%,#047857 100%);box-shadow:0 8px 26px rgba(16,185,129,.3)}
.sp{display:flex;flex-direction:column;gap:7px;min-width:0}
.sp .top{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:10.5px;font-weight:700;letter-spacing:.18em;color:var(--cyan)}
.sp .top b{font-family:'Space Grotesk','Inter',sans-serif;color:#fff;font-size:15px;letter-spacing:0}
.sp .top span span{color:var(--mute);font-weight:500;letter-spacing:.02em}
.sp .pre{display:flex;gap:5px;flex-wrap:wrap}
.sp .pre button{height:25px;padding:0 10px;border-radius:999px;border:1px solid var(--line);background:rgba(56,189,248,.06);font-size:11.5px;font-weight:600}
.sp .pre button:hover{border-color:var(--cyan)}
.sp .pre button.on{background:rgba(56,189,248,.24);border-color:var(--cyan);color:#fff}
input[type=range]{width:240px;accent-color:#38bdf8;cursor:pointer;margin:0}
#live{height:42px;padding:0 16px;border-radius:999px;border:1px solid rgba(148,163,184,.35);background:transparent;font-size:13px;font-weight:600;white-space:nowrap;transition:all .2s}
#live:hover{border-color:var(--cyan);background:rgba(56,189,248,.08)}
#pz{position:absolute;left:50%;top:18px;transform:translateX(-50%);padding:8px 18px;border-radius:999px;background:rgba(251,191,36,.14);border:1px solid rgba(251,191,36,.55);color:#fbbf24;font-size:11.5px;font-weight:700;letter-spacing:.18em;display:none;z-index:6}
#empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#94a3b8;z-index:4;text-align:center;padding:30px}
@media (max-width:900px){
 #app{flex-direction:column}
 #panel{width:100%;flex:0 0 40%;padding:16px;border-right:0;border-bottom:1px solid var(--line)}
 h1{font-size:28px}.big b{font-size:38px}#hud{display:none}
 input[type=range]{width:140px}
}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head><body>
<div id="app">
 <div id="panel">
  <div class="kick" id="t-kick"></div>
  <h1 id="t-title"></h1>
  <p class="desc" id="t-desc"></p>
  <div class="warn" id="warn"></div>
  <div class="big"><b id="total">0</b><span id="t-big"></span></div>
  <div class="sec" id="t-types"></div>
  <div id="rows"></div>
  <div class="sec" style="margin-top:22px" id="t-launch"></div>
  <div class="chips" id="chips"></div>
  <div class="help" id="t-help"></div>
 </div>
 <div id="stage">
  <canvas id="cv" tabindex="0"></canvas>
  <div id="clock"><span id="clk">--</span><i id="badge"></i></div>
  <div id="pz"></div>
  <div id="empty"></div>
  <div id="hud">
   <h4 id="h-title"></h4>
   <p><em id="l-name" style="font-style:normal"></em><span id="h-name" style="color:#7dd3fc"></span></p>
   <p><em id="l-id" style="font-style:normal"></em><span id="h-id">---</span></p>
   <p><em id="l-cls" style="font-style:normal"></em><span id="h-cls">---</span></p>
   <p><em id="l-alt" style="font-style:normal"></em><span id="h-alt" style="color:#34d399">---</span></p>
   <p><em id="l-vel" style="font-style:normal"></em><span id="h-vel">---</span></p>
   <p><em id="l-inc" style="font-style:normal"></em><span id="h-inc">---</span></p>
   <p><em id="l-per" style="font-style:normal"></em><span id="h-per">---</span></p>
   <p><em id="l-thr" style="font-style:normal"></em><span id="h-thr" style="color:#34d399">---</span></p>
  </div>
  <div id="bar">
   <button id="pp"></button>
   <div class="sp">
    <div class="top"><span id="t-speed"></span><span><b id="spv">120×</b> &nbsp;<span id="sph"></span></span></div>
    <input id="spr" type="range" min="0" max="3.7" step="0.01" value="2.08">
    <div class="pre" id="pre"></div>
   </div>
   <button id="live"></button>
  </div>
 </div>
</div>
<script>
(function(){
"use strict";
const D = __DATA__, L = __L__, FLAGGED = new Set(__FLAGS__);
const N = D.id.length;
const MU = 398600.4418, RE = 6378.137, J2 = 1.08262668e-3, TAU = Math.PI * 2, F = 1 / 298.257223563, DEG = 180 / Math.PI;
const $ = id => document.getElementById(id);

/* ---------- Metinler ---------- */
$('t-kick').textContent = L.kicker; $('t-title').textContent = L.title; $('t-desc').textContent = L.desc;
$('t-big').textContent = L.big; $('t-types').textContent = L.types; $('t-launch').textContent = L.launched;
$('t-help').textContent = L.help; $('t-speed').textContent = L.speed; $('live').textContent = '⟲ ' + L.live;
$('h-title').textContent = L.hud; $('pz').textContent = '⏸ ' + L.paused;
[['l-name','target'],['l-id','norad'],['l-cls','cls'],['l-alt','alt'],['l-vel','vel'],['l-inc','inc'],['l-per','per'],['l-thr','thr']].forEach(a => $(a[0]).textContent = L[a[1]] + ':');
$('h-name').textContent = L.hover;
if (N < 300) { const w = $('warn'); w.style.display = 'block'; w.textContent = L.few; }
if (N === 0) { const e = $('empty'); e.style.display = 'flex'; e.textContent = L.few; }
const fmt = n => n.toLocaleString(L.loc);

/* ---------- Yörünge elemanları ---------- */
const K = Uint8Array.from(D.k), AGE = Int32Array.from(D.age);
const A = new Float32Array(N), E = new Float32Array(N), INC = new Float32Array(N), SQ = new Float32Array(N);
const CI = new Float32Array(N), SI = new Float32Array(N);
const RA = new Float64Array(N), AP = new Float64Array(N), M0 = new Float64Array(N);
const NM = new Float64Array(N), DM = new Float64Array(N), DO = new Float64Array(N), DW = new Float64Array(N);
const PX = new Float32Array(N), PY = new Float32Array(N), PZ = new Float32Array(N);
const QX = new Float32Array(N), QY = new Float32Array(N), QZ = new Float32Array(N);
for (let i = 0; i < N; i++) {
  const o = i * 6, a = D.el[o], e = D.el[o+1], inc = D.el[o+2];
  A[i] = a; E[i] = e; INC[i] = inc; RA[i] = D.el[o+3]; AP[i] = D.el[o+4]; M0[i] = D.el[o+5];
  SQ[i] = Math.sqrt(1 - e * e); CI[i] = Math.cos(inc); SI[i] = Math.sin(inc);
  const ak = a * RE, n = Math.sqrt(MU / (ak * ak * ak)), p = ak * (1 - e * e), k = 1.5 * J2 * Math.pow(RE / p, 2), c = CI[i];
  NM[i] = n;
  DO[i] = -k * n * c;
  DW[i] = 0.5 * k * n * (5 * c * c - 1);
  DM[i] = n * (1 + 0.5 * k * SQ[i] * (3 * c * c - 1));
}
/* TEME (X,Y,Z) -> three.js (X, Z, -Y) */
function refresh(i, t) {
  const w = AP[i] + DW[i] * t, om = RA[i] + DO[i] * t;
  const cw = Math.cos(w), sw = Math.sin(w), cO = Math.cos(om), sO = Math.sin(om), ci = CI[i], si = SI[i];
  PX[i] = cO * cw - sO * sw * ci;      PY[i] = sw * si;       PZ[i] = -(sO * cw + cO * sw * ci);
  QX[i] = -cO * sw - sO * cw * ci;     QY[i] = cw * si;       QZ[i] = -(-sO * sw + cO * cw * ci);
}

/* ---------- Sahne ---------- */
const stage = $('stage'), cv = $('cv');
const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: false, powerPreference: 'high-performance' });
const PR = Math.min(window.devicePixelRatio || 1, 2);
renderer.setPixelRatio(PR); renderer.setClearColor(0x040817, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 400);
const th0 = D.gmst0 + 35 * Math.PI / 180, la0 = 22 * Math.PI / 180, dist0 = 3.3;
camera.position.set(dist0 * Math.cos(la0) * Math.cos(th0), dist0 * Math.sin(la0), -dist0 * Math.cos(la0) * Math.sin(th0));
const controls = new THREE.OrbitControls(camera, cv);
controls.enableDamping = true; controls.dampingFactor = 0.07; controls.enablePan = false;
controls.minDistance = 1.12; controls.maxDistance = 16; controls.rotateSpeed = 0.6; controls.zoomSpeed = 0.8;

const sunDir = new THREE.Vector3(D.sun[0], D.sun[1], D.sun[2]).normalize();
const sunL = new THREE.DirectionalLight(0xffffff, 1.9); sunL.position.copy(sunDir).multiplyScalar(20); scene.add(sunL);
scene.add(new THREE.AmbientLight(0x8fa3d1, 0.85));

const tl = new THREE.TextureLoader(); tl.setCrossOrigin('anonymous');
const earthMat = new THREE.MeshStandardMaterial({ color: 0x1a3f7a, roughness: 0.95, metalness: 0.0, emissive: 0x050b1c });
const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), earthMat); scene.add(earth);
tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg',
  t => { earthMat.map = t; earthMat.color.setHex(0xc9d3e6); earthMat.needsUpdate = true; }, undefined, () => {});
const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, depthWrite: false });
const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.008, 64, 48), cloudMat); scene.add(clouds);
tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png',
  t => { cloudMat.map = t; cloudMat.needsUpdate = true; }, undefined, () => {});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.07, 64, 48), new THREE.ShaderMaterial({
  transparent: true, side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false,
  vertexShader: 'varying vec3 vN; void main(){ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: 'varying vec3 vN; void main(){ float i = pow(max(0.0, 0.72 - dot(vN, vec3(0.0,0.0,1.0))), 3.0); gl_FragColor = vec4(0.32,0.62,1.0,1.0) * i; }'
})));

/* Yıldızlar */
{
  const n = 1400, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 2 - 1, a = Math.random() * TAU, s = Math.sqrt(1 - u * u), r = 150;
    pos[i*3] = r * s * Math.cos(a); pos[i*3+1] = r * u; pos[i*3+2] = r * s * Math.sin(a);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 1.3, sizeAttenuation: false, color: 0xa9b8d8, transparent: true, opacity: 0.65, depthWrite: false })));
}

/* ---------- Nesne noktaları (tek geometri, GPU'da renk/boyut/görünürlük) ---------- */
const P = new Float32Array(N * 3), COL = new Float32Array(N * 3), SIZE = new Float32Array(N), SHOW = new Float32Array(N);
const BASE = [2.6, 3.4, 3.6, 2.4, 8.0];
for (let i = 0; i < N; i++) {
  const c = L.colors[K[i]]; COL[i*3] = c[0]; COL[i*3+1] = c[1]; COL[i*3+2] = c[2];
  SIZE[i] = (BASE[K[i]] + (FLAGGED.has(D.n[i]) ? 3 : 0)) * PR;
}
const geo = new THREE.BufferGeometry();
const posAttr = new THREE.BufferAttribute(P, 3), showAttr = new THREE.BufferAttribute(SHOW, 1);
posAttr.setUsage(THREE.DynamicDrawUsage); showAttr.setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('position', posAttr); geo.setAttribute('aColor', new THREE.BufferAttribute(COL, 3));
geo.setAttribute('aSize', new THREE.BufferAttribute(SIZE, 1)); geo.setAttribute('aShow', showAttr);
const pts = new THREE.Points(geo, new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, depthTest: true,
  vertexShader: 'attribute vec3 aColor; attribute float aSize; attribute float aShow; varying vec3 vC;' +
    'void main(){ vC = aColor; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); gl_PointSize = aSize;' +
    'if (aShow < 0.5) { gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize = 0.0; } }',
  fragmentShader: 'varying vec3 vC; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;' +
    'float a = smoothstep(0.5, 0.12, d); gl_FragColor = vec4(vC, a * 0.96); }'
}));
pts.frustumCulled = false; pts.renderOrder = 2; scene.add(pts);

/* Seçili nesne halkası + yörünge çizgisi */
const ringCv = document.createElement('canvas'); ringCv.width = ringCv.height = 64;
{ const c = ringCv.getContext('2d'); c.strokeStyle = '#fff'; c.lineWidth = 5; c.beginPath(); c.arc(32, 32, 24, 0, TAU); c.stroke(); }
const ringGeo = new THREE.BufferGeometry(); ringGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
const ring = new THREE.Points(ringGeo, new THREE.PointsMaterial({ size: 30, sizeAttenuation: false, map: new THREE.CanvasTexture(ringCv), transparent: true, depthTest: false, depthWrite: false }));
ring.visible = false; ring.frustumCulled = false; ring.renderOrder = 5; scene.add(ring);
const SEG = 256, orbArr = new Float32Array((SEG + 1) * 3);
const orbGeo = new THREE.BufferGeometry(); orbGeo.setAttribute('position', new THREE.BufferAttribute(orbArr, 3));
const orbMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 });
const orbit = new THREE.Line(orbGeo, orbMat); orbit.visible = false; orbit.frustumCulled = false; orbit.renderOrder = 4; scene.add(orbit);

/* ---------- Filtre / sayaçlar ---------- */
const rowOf = k => k === 4 ? 0 : k;
const enabled = [false, true, true, true];       // çalışan uydular kapalı başlar
let launch = 'any', act = new Uint32Array(N), na = 0, counts = [0, 0, 0, 0], forceRef = true, dirty = true;
function ageOk(a) {
  switch (launch) {
    case 'any': return true;
    case 'd30': return a >= 0 && a <= 30;
    case 'y1':  return a >= 0 && a <= 365;
    case 'y5':  return a >= 0 && a <= 1826;
    case 'y10': return a >= 0 && a <= 3652;
    case 'old': return a > 3652;
  }
  return true;
}
const rowsEl = $('rows'), rowBtns = [];
['sat', 'dead', 'rb', 'deb'].forEach((key, r) => {
  const b = document.createElement('button'); b.className = 'row'; b.type = 'button';
  const c = L.colors[r], css = 'rgb(' + Math.round(c[0]*255) + ',' + Math.round(c[1]*255) + ',' + Math.round(c[2]*255) + ')';
  b.style.setProperty('--c', css);
  b.innerHTML = '<span class="dot"></span><span class="nm"></span><span class="ct">0</span>';
  b.querySelector('.nm').textContent = L[key];
  b.addEventListener('click', () => { enabled[r] = !enabled[r]; applyFilter(); });
  rowsEl.appendChild(b); rowBtns.push(b);
});
const chipsEl = $('chips'), chipBtns = {};
['any', 'd30', 'y1', 'y5', 'y10', 'old'].forEach(key => {
  const b = document.createElement('button'); b.className = 'chip'; b.type = 'button'; b.textContent = L.chips[key];
  b.addEventListener('click', () => { launch = key; applyFilter(); });
  chipsEl.appendChild(b); chipBtns[key] = b;
});
function updateUI() {
  rowBtns.forEach((b, r) => { b.classList.toggle('off', !enabled[r]); b.querySelector('.ct').textContent = fmt(counts[r]); });
  for (const k in chipBtns) chipBtns[k].classList.toggle('on', k === launch);
  let junk = 0; for (let r = 1; r < 4; r++) if (enabled[r]) junk += counts[r];
  $('total').textContent = fmt(junk);
}
function applyFilter() {
  counts = [0, 0, 0, 0]; na = 0;
  for (let i = 0; i < N; i++) {
    const r = rowOf(K[i]), ok = ageOk(AGE[i]);
    if (ok) counts[r]++;
    const show = ok && enabled[r];
    SHOW[i] = show ? 1 : 0;
    if (show) act[na++] = i;
  }
  showAttr.needsUpdate = true; forceRef = true; dirty = true;
  if (pinned >= 0 && !SHOW[pinned]) { pinned = -1; }
  if (hovered >= 0 && !SHOW[hovered]) { hovered = -1; }
  selectionChanged(); updateUI();
}

/* ---------- Zaman / hız ---------- */
let simT = (Date.now() - D.t0) / 1000, speed = 120, paused = false, tRef = -1e12;
const PRESETS = [1, 10, 60, 120, 300, 900, 3000];
const preEl = $('pre'), preBtns = [];
PRESETS.forEach(v => {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = v + '×';
  b.addEventListener('click', () => setSpeed(v)); preEl.appendChild(b); preBtns.push(b);
});
function dur(s) { return s < 60 ? s.toFixed(0) + ' ' + L.u[0] : s < 3600 ? (s / 60).toFixed(1) + ' ' + L.u[1] : (s / 3600).toFixed(1) + ' ' + L.u[2]; }
function setSpeed(v, fromSlider) {
  speed = Math.max(1, Math.min(5000, v));
  $('spv').textContent = Math.round(speed) + '×';
  $('sph').textContent = '· 1 ' + L.u[0] + ' = ' + dur(speed);
  if (!fromSlider) $('spr').value = Math.log10(speed);
  PRESETS.forEach((p, i) => preBtns[i].classList.toggle('on', Math.round(speed) === p));
}
function setPaused(p) {
  paused = p; const b = $('pp');
  b.textContent = paused ? '▶  ' + L.play : '⏸  ' + L.pause;
  b.classList.toggle('paused', paused);
  $('pz').style.display = paused ? 'block' : 'none';
}
$('pp').addEventListener('click', () => { setPaused(!paused); cv.focus(); });
$('spr').addEventListener('input', e => setSpeed(Math.pow(10, parseFloat(e.target.value)), true));
$('live').addEventListener('click', () => { simT = (Date.now() - D.t0) / 1000; setSpeed(1); setPaused(false); forceRef = true; dirty = true; });
window.addEventListener('keydown', e => {
  if (e.code === 'Space' || e.key === 'p' || e.key === 'P') { e.preventDefault(); setPaused(!paused); }
});
setSpeed(120); setPaused(false);

/* ---------- Konum hesabı ---------- */
function updatePositions(t) {
  const needRef = forceRef || Math.abs(t - tRef) > 120;
  if (needRef) { tRef = t; forceRef = false; }
  for (let k = 0; k < na; k++) {
    const i = act[k];
    if (needRef) refresh(i, t);
    let M = (M0[i] + DM[i] * t) % TAU; if (M < 0) M += TAU;
    const e = E[i];
    let Ea = M + e * Math.sin(M);
    if (e > 1e-4) {
      if (e > 0.8) Ea = Math.PI;
      for (let it = 0; it < 6; it++) { Ea -= (Ea - e * Math.sin(Ea) - M) / (1 - e * Math.cos(Ea)); }
    }
    const x = A[i] * (Math.cos(Ea) - e), y = A[i] * SQ[i] * Math.sin(Ea), j = i * 3;
    P[j] = PX[i] * x + QX[i] * y; P[j+1] = PY[i] * x + QY[i] * y; P[j+2] = PZ[i] * x + QZ[i] * y;
  }
  posAttr.needsUpdate = true;
}

/* ---------- Seçim (hover / sabitle) ---------- */
let hovered = -1, pinned = -1, mouse = { x: 0, y: 0, in: false, dirty: false }, down = null;
function selected() { return pinned >= 0 ? pinned : hovered; }
function selectionChanged() {
  const i = selected();
  if (i < 0) { orbit.visible = false; ring.visible = false; return; }
  refresh(i, simT);
  for (let s = 0; s <= SEG; s++) {
    const Ea = s / SEG * TAU, x = A[i] * (Math.cos(Ea) - E[i]), y = A[i] * SQ[i] * Math.sin(Ea);
    orbArr[s*3] = PX[i] * x + QX[i] * y; orbArr[s*3+1] = PY[i] * x + QY[i] * y; orbArr[s*3+2] = PZ[i] * x + QZ[i] * y;
  }
  orbGeo.attributes.position.needsUpdate = true;
  const c = L.colors[K[i]]; orbMat.color.setRGB(c[0], c[1], c[2]);
  orbit.visible = true; ring.visible = true;
}
function occluded(px, py, pz) {
  const c = camera.position, dx = px - c.x, dy = py - c.y, dz = pz - c.z, dd = dx*dx + dy*dy + dz*dz;
  const t = -(c.x*dx + c.y*dy + c.z*dz) / dd; if (t <= 0 || t >= 1) return false;
  const qx = c.x + dx*t, qy = c.y + dy*t, qz = c.z + dz*t; return qx*qx + qy*qy + qz*qz < 1.0;
}
const tv = new THREE.Vector3();
function pick(mx, my) {
  const w = cv.clientWidth, h = cv.clientHeight; let best = -1, bd = 100;
  camera.updateMatrixWorld();
  for (let k = 0; k < na; k++) {
    const i = act[k], j = i * 3;
    tv.set(P[j], P[j+1], P[j+2]).project(camera);
    if (tv.z > 1 || tv.z < -1) continue;
    const sx = (tv.x * 0.5 + 0.5) * w - mx, sy = (-tv.y * 0.5 + 0.5) * h - my, d = sx*sx + sy*sy;
    if (d < bd && !occluded(P[j], P[j+1], P[j+2])) { bd = d; best = i; }
  }
  return best;
}
cv.addEventListener('pointermove', e => {
  const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; mouse.in = true; mouse.dirty = true;
});
cv.addEventListener('pointerleave', () => { mouse.in = false; if (hovered >= 0) { hovered = -1; selectionChanged(); } });
cv.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; cv.focus(); });
cv.addEventListener('pointerup', e => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); down = null;
  if (moved > 5) return;
  const r = cv.getBoundingClientRect(), i = pick(e.clientX - r.left, e.clientY - r.top);
  pinned = (i >= 0 && i !== pinned) ? i : -1; selectionChanged();
});

/* ---------- HUD ---------- */
const clsName = k => [L.cSat, L.cDead, L.cRb, L.cDeb, L.cStn][k];
function updateHUD() {
  const i = selected();
  if (i < 0) {
    $('h-name').textContent = L.hover; ['h-id','h-cls','h-alt','h-vel','h-inc','h-per','h-thr'].forEach(id => $(id).textContent = '---');
    return;
  }
  const j = i * 3, r = Math.hypot(P[j], P[j+1], P[j+2]);
  const alt = (r - (1 - F * Math.pow(P[j+1] / r, 2))) * RE;
  const v = Math.sqrt(MU * (2 / (r * RE) - 1 / (A[i] * RE)));
  $('h-name').textContent = D.n[i] + (pinned === i ? ' 📌' : '');
  $('h-id').textContent = D.id[i]; $('h-cls').textContent = clsName(K[i]);
  $('h-alt').textContent = alt.toFixed(0) + ' km'; $('h-vel').textContent = v.toFixed(2) + ' km/s';
  $('h-inc').textContent = (INC[i] * DEG).toFixed(2) + '°'; $('h-per').textContent = (TAU / NM[i] / 60).toFixed(1) + ' ' + L.u[1];
  const fl = FLAGGED.has(D.n[i]), t = $('h-thr'); t.textContent = fl ? L.flag : L.noflag; t.style.color = fl ? '#fc3d21' : '#34d399';
  ring.geometry.attributes.position.array.set([P[j], P[j+1], P[j+2]]); ring.geometry.attributes.position.needsUpdate = true;
}

/* ---------- Boyut ---------- */
function resize() {
  const w = Math.max(10, stage.clientWidth), h = Math.max(10, stage.clientHeight);
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
if (window.ResizeObserver) new ResizeObserver(resize).observe(stage); else window.addEventListener('resize', resize);
resize();

/* ---------- Döngü ---------- */
let last = performance.now(), fr = 0;
applyFilter();
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, Math.max(0, (now - last) / 1000)); last = now;
  if (!paused) simT += dt * speed;
  if (!paused || dirty) {
    updatePositions(simT); dirty = false;
    earth.rotation.y = D.gmst0 + D.omega * simT; clouds.rotation.y = earth.rotation.y * 1.0 + 0.0004 * simT / 3600;
  }
  if (mouse.dirty && mouse.in && !down) {
    mouse.dirty = false;
    const i = pick(mouse.x, mouse.y);
    if (i !== hovered) { hovered = i; selectionChanged(); cv.style.cursor = i >= 0 ? 'pointer' : 'grab'; }
  }
  if ((fr++ & 3) === 0) {
    updateHUD();
    const drift = Math.abs(simT - (Date.now() - D.t0) / 1000);
    $('clk').textContent = new Date(D.t0 + simT * 1000).toISOString().slice(0, 19).replace('T', ' ') + ' UTC';
    const bd = $('badge');
    bd.className = paused ? 'pz' : (speed === 1 && drift < 3 ? '' : 'sim');
    bd.textContent = paused ? L.paused : (speed === 1 && drift < 3 ? '● ' + L.livebadge : L.sim);
  }
  if (selected() >= 0 && paused) updateHUD();
  controls.update(); renderer.render(scene, camera);
}
cv.style.cursor = 'grab';
requestAnimationFrame(animate);
})();
</script></body></html>
"""
    html_tracker = (TRACKER_HTML.replace("__L__", json.dumps(_L, ensure_ascii=False))
                    .replace("__FLAGS__", json.dumps(flags, ensure_ascii=False))
                    .replace("__DATA__", json.dumps(tracker, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
    embed_html(html_tracker, height=820)

# =====================================================================
# 3. MODÜL: COLLISION RISK
# =====================================================================
elif st.session_state.active_page == PAGES[2]:
    st.markdown(T("### 🚨 KRİTİK ÇARPIŞMA TEHDİT MATRİSİ", "### 🚨 CRITICAL COLLISION THREAT MATRIX"))

    def tles_for(choice):
        rng = np.random.default_rng(11)
        act, deb = CAT["groups"]["active"], CAT["debris"]

        def pick(lst, n):
            if not lst:
                return []
            return [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)]

        if choice.startswith("Active ("):
            return tuple(pick(act, 1200))
        if choice.startswith("Debris"):
            return tuple(pick(deb, 1200))
        return tuple(pick(act, 600) + pick(deb, 600))

    @st.cache_data(ttl=1800, show_spinner=False)
    def screen_conjunctions(tles, hours, thr, step_s=30):
        if len(tles) < 2:
            return pd.DataFrame()
        sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
        names = [t[0] for t in tles]
        ids = [norad_id(t[1]) for t in tles]
        arr = SatrecArray(sats)
        t0 = utc_now()
        n = int(hours * 3600 / step_s)
        best = {}
        # 1) Kaba tarama: her adımda KD-Tree ile eşik altındaki çiftler
        for c0 in range(0, n, 240):
            times = [t0 + timedelta(seconds=step_s * k) for k in range(c0, min(n, c0 + 240))]
            e, r, v = arr.sgp4(*to_jd_arrays(times))
            for j, t in enumerate(times):
                ok = np.where(e[:, j] == 0)[0]
                if len(ok) < 2:
                    continue
                p = r[ok, j, :]
                for a, b in cKDTree(p).query_pairs(thr):
                    i1, i2 = int(ok[a]), int(ok[b])
                    # Aynı yörüngede birlikte uçan nesneleri (ör. kenetli modüller) atla
                    if np.linalg.norm(v[i1, j] - v[i2, j]) < 0.5:
                        continue
                    d = float(np.linalg.norm(p[a] - p[b]))
                    if (i1, i2) not in best or d < best[(i1, i2)][0]:
                        best[(i1, i2)] = (d, t)
        # 2) İnce tarama: en yakın 40 çift için ±35 s pencerede 0.25 s adımla TCA
        top = sorted(best.items(), key=lambda kv: kv[1][0])[:40]
        offs = np.arange(-35, 35.01, 0.25)
        rows = []
        for (i1, i2), (_, t) in top:
            bd, brv, bt = math.inf, 0.0, t
            for o in offs:
                tt = t + timedelta(seconds=float(o))
                jd, fr = to_jd(tt)
                ea, ra, va = sats[i1].sgp4(jd, fr)
                eb, rb, vb = sats[i2].sgp4(jd, fr)
                if ea or eb:
                    continue
                d = math.dist(ra, rb)
                if d < bd:
                    bd, brv, bt = d, math.dist(va, vb), tt
            if math.isfinite(bd):
                rows.append({
                    "Object A": f"{names[i1]} ({ids[i1]})", "Object B": f"{names[i2]} ({ids[i2]})",
                    "Miss Distance (km)": round(bd, 3), "Rel. Velocity (km/s)": round(brv, 2),
                    "TCA (UTC)": bt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Risk": ("🔴 CRITICAL" if bd < 1 else "🟠 HIGH" if bd < 5 else "🟡 WATCH" if bd < 25 else "⚪ SCREENED"),
                    "_a": names[i1], "_b": names[i2],
                })
        df = pd.DataFrame(rows)
        return df.sort_values("Miss Distance (km)").reset_index(drop=True) if not df.empty else df

    c1, c2, c3 = st.columns(3)
    _CH = ["Active (random 1200)", "Debris fields (random 1200)", "Active × Debris (600 + 600)"]
    _CH_TR = {_CH[0]: "Aktif (rastgele 1200)", _CH[1]: "Enkaz alanları (rastgele 1200)", _CH[2]: "Aktif × Enkaz (600 + 600)"}
    choice = c1.selectbox(T("Taranacak katalog", "Catalog to screen"), _CH, index=2, key="sg_choice",
                          format_func=lambda x: _CH_TR[x] if st.session_state.lang == "tr" else x)
    hours = c2.slider(T("İleri bakış (saat)", "Look-ahead (hours)"), 1, 24, 12, key="sg_hours")
    thr = c3.slider(T("Kaba tarama eşiği (km)", "Coarse screening threshold (km)"), 20, 120, 60, key="sg_thr")

    if st.button(T("🛰️ YAKINLAŞMA TARAMASINI BAŞLAT", "🛰️ RUN CONJUNCTION SCREENING"), type="primary", **stretch(st.button)):
        with st.spinner(T("SGP4 propagasyonu + KD-Tree taraması çalışıyor...", "Running SGP4 propagation + KD-Tree screening...")):
            df = screen_conjunctions(tles_for(choice), hours, float(thr))
        st.session_state.conj_df = df
        if df.empty:
            st.session_state.conj_meta = {"count": 0, "names": set()}
            st.session_state.defense_active = False
        else:
            flagged = df[df["Miss Distance (km)"] < 25]
            st.session_state.conj_meta = {"count": len(df), "names": set(flagged["_a"]) | set(flagged["_b"])}
            st.session_state.defense_active = bool((df["Miss Distance (km)"] < 5).any())

    df = st.session_state.conj_df
    if df is None:
        st.info(T("Taramayı başlatmak için butona bas.", "Press the button to start the screening."))
    elif df.empty:
        st.success(T("Seçilen pencerede yaklaşma bulunamadı.", "No close approaches found in the selected window."))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T("Yakın geçişler", "Close approaches"), len(df))
        m2.metric(T("En yakın geçiş mesafesi", "Closest miss distance"), f"{df['Miss Distance (km)'].min():.2f} km")
        m3.metric("< 5 km", int((df["Miss Distance (km)"] < 5).sum()))
        m4.metric(T("En yüksek göreli hız", "Highest rel. velocity"), f"{df['Rel. Velocity (km/s)'].max():.1f} km/s")
        view = df.drop(columns=["_a", "_b"])
        st.dataframe(_disp(view), hide_index=True, **stretch(st.dataframe))

    # --- OrbitRisk (orbitrisk.py) ---
    st.markdown("<hr style='border: 1px solid rgba(56,189,248,0.25); margin: 25px 0;'>", unsafe_allow_html=True)
    # Koyu Tema Özel CSS Styling
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] { color: #06b6d4 !important; font-family: monospace; }
        .card {
            background-color: #111827;
            border: 1px solid #1f2937;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER / ÜST BİLGİ ---
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title(T("🛰️ ORBITRISK B2B SaaS Terminali", "🛰️ ORBITRISK B2B SaaS Terminal"))
        st.caption(T("Uzay Ekonomisi Finansal Risk ve Karar Destek Sistemi (NASA OTPS Uyumlu)",
                     "Space Economy Financial Risk & Decision Support System (NASA OTPS Aligned)"))

    with col_head2:
        st.metric(label=T("Toplam Takip Edilen Değer (LEO/GEO)", "Total Tracked Value (LEO/GEO)"),
                  value=T("$954,0 Milyar", "$954.0 Billion"))

    st.divider()

    # --- RED ALERT SİMÜLASYONU DURUMU ---
    if "red_alert" not in st.session_state:
        st.session_state.red_alert = False

    def trigger_alert():
        st.session_state.red_alert = True

    # --- ANA DÜZEN (LEFT & RIGHT COLUMNS) ---
    left_col, right_col = st.columns([2, 1])

    # ===============================
    # SOL SÜTUN: METRİKLER & GRAFİK
    # ===============================
    with left_col:
        # 3'lü Özet Kartları
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("10 YILLIK FİNANSAL RİSK HAVUZU", "10-YR FINANCIAL RISK POOL")}</span>
                    <h3 style="color:white; margin:5px 0;">{T("25,8 – 42,3 Milyar &#36;", "&#36;25.8B – &#36;42.3B")}</h3>
                    <span style="color:#ef4444; font-size:11px;">{T("📈 Gizli operatör maliyet maruziyeti", "📈 Hidden operator cost exposure")}</span>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("OPTİMİZE SENARYO KAYBI", "OPTIMIZED SCENARIO LOSS")}</span>
                    <h3 style="color:#06b6d4; margin:5px 0;">{T("14,2 Milyar &#36;", "&#36;14.2 Billion")}</h3>
                    <span style="color:#10b981; font-size:11px;">{T("🛡️ Erken uyarı ile 16,5 Milyar &#36; tasarruf", "🛡️ &#36;16.5B saved via early warning")}</span>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("GÜVENLİK KÜRESİ EŞİĞİ", "SAFETY SPHERE THRESHOLD")}</span>
                    <h3 style="color:#f59e0b; margin:5px 0;">&lt; 50.0 km</h3>
                    <span style="color:#f59e0b; font-size:11px;">{T("🔔 Aktif 3D Yakınlaşma Matrisi", "🔔 Active 3D Conjunction Matrix")}</span>
                </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Financial Breakdown Grafiği (Plotly)
        st.subheader(T("Yörünge Riski Finansal Dökümü (10 Yıllık Ufuk)", "Orbital Risk Financial Breakdown (10-Year Horizon)"))
        st.caption(T("Temel Maruziyet ile Optimize SaaS Müdahalesinin Karşılaştırması (NASA OTPS Modeli)",
                     "Comparing Baseline Exposure vs. Optimized SaaS Intervention (NASA OTPS Model)"))

        categories = [T('Toplam Ekonomik Etki', 'Total Economic Impact'), T('Anomali Riski', 'Anomaly Risk'),
                      T('Toplam Kayıp Riski', 'Total Loss Risk'), T('Manevra Yakıt Maliyeti', 'Maneuver Fuel Cost')]
        baseline_data = [42.3, 30.7, 11.1, 0.56]
        optimized_data = [14.2, 9.1, 3.4, 0.18]

        fig = go.Figure(data=[
            go.Bar(name=T('Temel Maruziyet (Milyar $)', 'Baseline Exposure ($B)'), x=categories, y=baseline_data, marker_color='#ef4444'),
            go.Bar(name=T('Optimize SaaS Senaryosu (Milyar $)', 'Optimized SaaS Scenario ($B)'), x=categories, y=optimized_data, marker_color='#06b6d4')
        ])
        fig.update_layout(
            barmode='group',
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            font=dict(color='#9ca3af'),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, **stretch(st.plotly_chart))

        # Canlı Yakınlaşma / Conjunction Tablosu
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.subheader(T("🌐 Aktif Yakınlaşmalar ve Güvenlik Küresi Durumu", "🌐 Active Conjunctions & Safety Sphere Status"))
        with col_t2:
            if st.session_state.red_alert:
                st.error(T("KIRMIZI ALARM: Güvenlik Küresi İhlal Edildi (<50 km)!", "RED ALARM: Safety Sphere Breached (<50km)!"))
            else:
                st.success(T("Tüm Sistemler Normal", "All Systems Nominal"))

        # Tablo Verisi
        k_sat, k_deb, k_app, k_kin, k_score, k_stat = (
            T("Hedef Uydu", "Target Satellite"), T("Enkaz / Nesne Kimliği", "Debris / Object ID"),
            T("En Yakın Geçiş", "Closest Approach"), T("Kinetik Potansiyel", "Kinetic Potential"),
            T("Dinamik Risk Skoru", "Dynamic Risk Score"), T("Durum", "Status"))
        table_data = [
            {k_sat: "Sat-Alpha (LEO 650km)", k_deb: "COSMOS 2251 Frg", k_app: T("18,4 km", "18.4 km"),
             k_kin: T("34,2 J/g (Yıkıcı Değil)", "34.2 J/g (Non-Catastrophic)"), k_score: "64 / 100", k_stat: T("Uyarı", "Warning")},
            {k_sat: "IRIDIUM-97 Fleet", k_deb: T("Roket Gövdesi (SL-8 R/B)", "Rocket Body (SL-8 R/B)"), k_app: T("72,1 km", "72.1 km"),
             k_kin: T("12,1 J/g", "12.1 J/g"), k_score: "22 / 100", k_stat: T("Temiz", "Clear")}
        ]

        # Red Alert tetiklendiyse yeni satır ekle
        if st.session_state.red_alert:
            table_data.insert(0, {
                k_sat: "Sat-Omega (LEO 800km)",
                k_deb: "IRIDIUM 33 Frg",
                k_app: T("11,2 km (KRİTİK)", "11.2 km (CRITICAL)"),
                k_kin: T("48,6 J/g (Yıkıcı)", "48.6 J/g (Catastrophic)"),
                k_score: "92 / 100",
                k_stat: T("KIRMIZI ALARM", "RED ALARM")
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, hide_index=True, **stretch(st.dataframe))


    # ===============================
    # SAĞ SÜTUN: HESAPLAMA VE FİYATLANDIRMA
    # ===============================
    with right_col:
        # Alarm Butonu
        st.button(T("🚨 Kırmızı Alarmı Simüle Et", "🚨 Simulate Red Alarm"), on_click=trigger_alert, type="primary", **stretch(st.button))

        st.write("")

        # InsurTech Policy Simulator
        st.subheader(T("🧮 InsurTech Poliçe Simülatörü", "🧮 InsurTech Policy Simulator"))
        st.caption(T("Uydu kütlesi ve yörünge yoğunluğuna göre dinamik prim ayarlarını hesaplayın.",
                     "Calculate dynamic premium adjustments based on satellite mass and orbit density."))

        sats = st.slider(T("Aktif Takımyıldız Büyüklüğü (Uydu):", "Active Constellation Size (Satellites):"),
                         min_value=10, max_value=1000, value=120, key="sg_sats")

        rocket_factor = st.selectbox(
            T("Yüksek Kütleli Roket Gövdesi Faktörü:", "High-Mass Rocket Body Factor:"),
            options=[1.0, 1.5], key="sg_rocket",
            format_func=lambda x: (T("Standart Konfigürasyon (1.0x)", "Standard Configuration (1.0x)") if x == 1.0
                                   else T("Ağır Roket Gövdesi Yakınlığı (1.5x)", "Heavy Rocket Body Proximity (1.5x)")),
            index=1
        )

        density_factor = st.selectbox(
            T("Yörünge Malzeme Yoğunluğu Çarpanı:", "Orbital Material Density Multiplier:"),
            options=[1.0, 1.25], key="sg_density",
            format_func=lambda x: (T("Düşük Yoğunluklu Enkaz Alanı (1.0x)", "Low Density Debris Field (1.0x)") if x == 1.0
                                   else T("Yüksek Yoğunluklu Parça Kümesi (1.25x)", "High Density Fragment Cluster (1.25x)")),
            index=1
        )

        # Hesaplama Algoritması
        base_score = min(98, round((sats / 10) * 0.4 + (rocket_factor * 20) + (density_factor * 15)))
        estimated_cost = (sats * 18500 * rocket_factor * density_factor) / 1000000

        st.markdown(f"""
            <div style="background-color: #1f2937; padding: 12px; border-radius: 8px; margin-top: 10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#9ca3af; font-size:12px;">{T("Dinamik Risk Skoru:", "Dynamic Risk Score:")}</span>
                    <span style="color:#f59e0b; font-weight:bold; font-family:monospace;">{base_score} / 100</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#9ca3af; font-size:12px;">{T("Tah. Yıllık Poliçe Riski:", "Est. Annual Policy Risk:")}</span>
                    <span style="color:#06b6d4; font-weight:bold; font-family:monospace; font-size:18px;">${estimated_cost:.2f}M</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.write("")

        # B2B SaaS Tiers
        st.subheader(T("📦 B2B SaaS Paketleri", "📦 B2B SaaS Tiers"))

        st.markdown(f"""
            <div class="card" style="border-left: 3px solid #3b82f6;">
                <strong style="color:white; font-size:12px;">BASIC / STANDART</strong> — <span style="color:#06b6d4; font-size:11px;">{T("Sabit Ücret / Abonelik", "Flat-Rate / Sub")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Küçük Takımyıldız Operatörleri ve Üniversite Araştırmacıları. Manevra ve yakıt optimize edici.", "Target: Small Constellation Operators & University Researchers. Maneuver & fuel optimizer.")}</p>
            </div>
            <div class="card" style="border-left: 3px solid #06b6d4; background-color: #0e2433;">
                <span style="background-color:#06b6d4; color:#0a0f1d; font-size:9px; font-weight:bold; padding:2px 6px; border-radius:10px; float:right;">{T("POPÜLER", "POPULAR")}</span>
                <strong style="color:white; font-size:12px;">PRO / ENTERPRISE</strong> — <span style="color:#06b6d4; font-size:11px;">{T("Kullanıma Dayalı API", "Usage-Based API")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Ticari Uydu Filo Yöneticileri. Gerçek zamanlı anomali tespiti ve kesinti maliyeti analizi.", "Target: Commercial Satellite Fleet Managers. Real-time anomaly detection & outage cost analysis.")}</p>
            </div>
            <div class="card" style="border-left: 3px solid #a855f7;">
                <strong style="color:white; font-size:12px;">INSTITUTIONAL / INSURTECH</strong> — <span style="color:#a855f7; font-size:11px;">{T("Yıllık Sözleşme", "Annual Contract")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Uzay Savunma Ajansları ve Sigortacılar. Kümülatif risk modelleme ve özel poliçe fiyatlandırma matrisi.", "Target: Space Defense Agencies & Insurers. Cumulative risk modeling & custom policy pricing matrix.")}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.divider()
    st.caption(T("OrbitRisk Finansal Terminali • NASA OTPS ve SpaceApps Veri Çerçeveleriyle Entegre • Akademik ve B2B Karar Destek Platformu",
                 "OrbitRisk Financial Terminal • Integrated with NASA OTPS & SpaceApps Data Frameworks • Academic & B2B Decision Support Platform"))

# =====================================================================
# 4. MODÜL: RE-ENTRY
# =====================================================================
elif st.session_state.active_page == PAGES[3]:
    st.markdown(T("### 🔥 SPACE GRAVE [SP4C3 GR4V3] Atmosfere Yeniden Giriş ve Parçalanma Simülatörü",
                  "### 🔥 SPACE GRAVE [SP4C3 GR4V3] Atmospheric Re-Entry & Disintegration Simulator"))
    st.markdown(f"<p style='color:#94a3b8;font-size:13px;'>{T('Üstel atmosfer + balistik sürükleme + Sutton-Graves ısı akısı + çok parçalı kırılma + kırmızı/sarı/yeşil bölge (yer etkisi) analizi.', 'Exponential atmosphere + ballistic drag + Sutton-Graves heating + multi-fragment break-up + red/amber/green ground-hazard zone analysis.')}</p>", unsafe_allow_html=True)
    st.markdown("---")

    sat_names = df_api["Satellite Name"].tolist() if not df_api.empty else []
    st.selectbox(T("Hedef Enkaz / Uydu:", "Target Debris / Satellite:"), ["Custom object"] + sat_names, key="sg_target",
                 format_func=lambda x: T("Özel nesne", "Custom object") if x == "Custom object" else x)
    s1, s2, s3 = st.columns(3)
    s1.slider(T("Nesne Kütlesi (kg)", "Object Mass (kg)"), 50, 5000, 1200, key="sg_mass")
    s2.slider(T("Kesit alanı (m²)", "Cross-section area (m²)"), 0.1, 30.0, 4.0, key="sg_area")
    s3.slider(T("120 km'de uçuş yolu açısı (°)", "Flight-path angle at 120 km (°)"), 0.5, 45.0, 3.0, key="sg_angle")
    st.selectbox(T("Baskın yapısal malzeme", "Dominant structural material"), list(MATERIALS.keys()), key="sg_material", format_func=_mat_name)
    _rk = _sim_risk(st.session_state.get("sg_target", "Custom object"))
    st.info(T(f"Risk analizi bağlantısı: {'tarama sonucu bulundu (' + str(_rk['n']) + ' yakın geçiş)' if _rk['n'] is not None else 'henüz tarama yok'}; hedefin TLE yörüngesi simülasyona otomatik aktarılır.",
              f"Risk-analysis link: {'scan found (' + str(_rk['n']) + ' close approaches)' if _rk['n'] is not None else 'no scan yet'}; the target's TLE orbit is passed to the simulation automatically."))
    st.button(T("☄️ SPACE GRAVE YÖRÜNGEDEN ÇIKARMA SEKANSINI BAŞLAT", "☄️ INITIATE SPACE GRAVE DE-ORBIT SEQUENCE"),
              type="primary", on_click=_sim_open, key="sg_sim_start", **stretch(st.button))

# =====================================================================
# SAĞ ALT: MİNİ SORU-CEVAP BOTU (kural tabanlı; yapay zeka kullanmaz)
# =====================================================================
BOT_PLACEHOLDER = None  # dile göre _bot_ph() üretir


def _bot_conj():
    m = st.session_state.conj_meta
    if m:
        return T(f"{m['count']} yaklaşma", f"{m['count']} close approaches")
    return T("henüz tarama yapılmadı", "no screening run yet")


def _bot_defense():
    return _defense_txt()


# id: (butonda görünen soru, cevap üreten fonksiyon) — cevaplar sitedeki gerçek davranışa göre yazıldı
BOT_QA_TR = {
    # --- Genel ---
    "site": ("Bu site ne yapıyor?", lambda c: (
        "<b>SPACE GRAVE</b>, NASA Space Apps için hazırlanmış bir uzay çöpü komuta merkezi. "
        "CelesTrak'teki gerçek yörünge verilerini (TLE) SGP4 ile hesaplıyor. "
        f"Şu an <b>{c['n_active']:,}</b> aktif uydu ve <b>{c['n_debris']:,}</b> enkaz parçası izleniyor.")),
    "pages": ("Hangi sayfalar var?", lambda c: (
        "Dört modül var: <b>Genel Durum Özeti</b> (özet kartlar ve grafik), "
        "<b>3D Dünya ve Yörünge Radarı</b> (canlı 3D küre), <b>Çarpışma Riski Analizi</b> "
        "(yakınlaşma taraması + OrbitRisk mali panel) ve <b>Yeniden Giriş ve Yanma Sim.</b> "
        "(atmosfere giriş simülasyonu). Üstteki butonlarla geçiş yapabilirsin.")),
    "veri": ("Veriler nereden geliyor?", lambda c: (
        "Veriler <b>CelesTrak</b>'ten çekiliyor: aktif uydular, görünür uydular ve Cosmos 2251, "
        "Iridium 33, Fengyun-1C, Cosmos 1408 enkaz grupları. Veri 2 saat önbellekte tutulur. "
        f"Son güncelleme: <b>{c['fetched']}</b>. Bağlantı yoksa ISS, Hubble ve Tiangong yedek verisi kullanılır.")),
    "feed": ("Veri durumu ne demek?", lambda c: (
        "<b>LIVE 🟢</b> canlı veri, <b>CACHED 🟡</b> önbellek, <b>PARTIAL 🟡</b> bazı gruplar alınamadı, "
        f"<b>OFFLINE 🟠</b> yedek veri demek. Şu an: <b>{c['feed']}</b>.")),
    "sgp4": ("SGP4 nedir?", lambda c: (
        "<b>SGP4</b>, TLE satırlarından bir nesnenin belirli bir andaki konum ve hızını hesaplayan "
        "standart yörünge modelidir. Radar, çarpışma taraması ve tablo değerleri bununla üretiliyor.")),
    "kessler": ("Kessler Sendromu nedir?", lambda c: (
        "<b>Kessler Sendromu</b>: yörüngedeki enkaz yoğunluğu arttıkça çarpışmaların yeni parçalar "
        "üretmesi, bunların da yeni çarpışmalara yol açması zinciridir. Site, çarpışma taraması ve "
        "enkaz grafikleriyle bu riski görünür kılıyor.")),
    # --- Overview ---
    "cards": ("Kartlar ne anlama geliyor?", lambda c: (
        "<b>Katalog Nesneleri</b> aktif + enkaz toplamı, <b>Enkaz Parçaları</b> enkaz sayısı, "
        "<b>Aktif Uydular</b> aktif uydu sayısı. <b>Yakın Geçişler</b> Çarpışma sayfasında tarama "
        f"yapınca dolar (şu an: {_bot_conj()}). <b>Savunma Matrisi</b> şu an: <b>{_bot_defense()}</b>.")),
    "shells": ("Yörünge grafiği ne gösteriyor?", lambda c: (
        "Nesnelerin <b>irtifa × eğim</b> dağılımını: mavi noktalar aktif uydular, kırmızılar enkaz. "
        "Aynı yerde kümelenen noktalar aynı yörünge kabuğunu paylaşır. Okunabilirlik için rastgele "
        "örneklem ve 2200 km altı gösterilir.")),
    "table": ("Canlı tabloyu açıkla", lambda c: (
        "Tablo 'visual' grubundaki uyduları listeler; irtifa, eğim, periyot ve hız SGP4 ile anlık "
        "hesaplanır. ISS, CSS ve Tiangong <b>MÜRETTEBATLI</b> olarak işaretlenir.")),
    "decaying": ("ALÇALIYOR ne demek?", lambda c: (
        "<b>ALÇALIYOR ⚠️</b>, irtifası 300 km'nin altına inen nesneler için kullanılan etikettir. "
        "Bu yükseklikte atmosfer sürtünmesi yörüngeyi hızla düşürür.")),
    # --- 3D Radar ---
    "radar_use": ("Radarı nasıl kullanırım?", lambda c: (
        "Küreyi sürükleyerek döndür, tekerlekle yakınlaştır. Bir nesnenin üstüne gelince yörünge "
        "çizgisi belirir ve sağdaki panelde isim, NORAD ID, irtifa ve hız görünür. Sol üstteki "
        "seçici zaman hızını (60×–900×) değiştirir.")),
    "radar_colors": ("Renkler ne demek?", lambda c: (
        "Açık gri gövdeli küçük uydular aktif uydulardır, <b>yeşil</b> olan ISS, <b>kırmızı</b> parçalar "
        "enkazdır. Etrafında kırmızı halka olan nesne, çarpışma taramasında 25 km altında yaklaşma "
        "yapmıştır (panelde İŞARETLİ yazar).")),
    "radar_calc": ("Konumlar nasıl hesaplanıyor?", lambda c: (
        "Her nesne için gerçek TLE'den SGP4 ile 100 adımlık (60 sn aralıklı) bir yörünge hesaplanır; "
        "tarayıcıda akıcı hareket için ara değerler üretilir. Hareket yaklaşık 99 dakikada bir başa sarar.")),
    "radar_limit": ("Neden hepsi gösterilmiyor?", lambda c: (
        "Performans için slider'larla en fazla 400 aktif uydu ve 600 enkaz parçası (rastgele seçilmiş) "
        "gösterilir. 'visual' grubundaki uydular her zaman eklenir.")),
    # --- Collision ---
    "col_scan": ("Tarama nasıl çalışıyor?", lambda c: (
        "İki aşamalı: önce 30 sn adımlarla KD-Tree ile eşik mesafenin altındaki çiftler bulunur; sonra "
        "en yakın 40 çift için ±35 sn pencerede 0,25 sn adımla en yakın geçiş anı (TCA) hesaplanır. "
        "Birlikte uçan nesneler elenir.")),
    "col_risk": ("Risk seviyeleri neler?", lambda c: (
        "Geçiş mesafesine göre: <b>&lt; 1 km</b> 🔴 KRİTİK, <b>&lt; 5 km</b> 🟠 YÜKSEK, "
        "<b>&lt; 25 km</b> 🟡 İZLEME, daha uzağı ⚪ ELENDİ.")),
    "col_def": ("Savunma Matrisi nedir?", lambda c: (
        "5 km altında en az bir yaklaşma bulunursa <b>Savunma Matrisi</b> AKTİF olur (Genel Durum "
        f"kartında görünür). Şu an: <b>{_bot_defense()}</b>.")),
    "col_tca": ("Geçiş mesafesi ve TCA nedir?", lambda c: (
        "<b>Geçiş Mesafesi</b> iki nesnenin en yakın geçişteki uzaklığı, <b>TCA</b> bu geçişin UTC zamanı, "
        "<b>Göreli Hız</b> nesnelerin birbirine göre hızıdır. Göreli hız yükseldikçe çarpışmanın enerjisi de büyür.")),
    "col_red": ("Kırmızı Alarm ne yapar?", lambda c: (
        "Sayfanın altındaki OrbitRisk panelinde <b>🚨 Kırmızı Alarmı Simüle Et</b> demo senaryosunu başlatır: "
        "Sat-Omega için 11,2 km'lik kritik bir yaklaşma tabloya eklenir ve 50 km'lik güvenlik küresi "
        "durumu KIRMIZI ALARM'a geçer.")),
    "col_policy": ("Poliçe hesabı nasıl?", lambda c: (
        "<b>Risk Skoru</b> = min(98, uydu sayısı/10 × 0,4 + roket faktörü × 20 + yoğunluk faktörü × 15). "
        "<b>Yıllık poliçe riski</b> = uydu sayısı × 18.500 × roket faktörü × yoğunluk faktörü "
        "(milyon dolar olarak gösterilir).")),
    "col_fin": ("OrbitRisk paneli nedir?", lambda c: (
        "Çarpışma sayfasının alt bölümü <b>OrbitRisk</b> mali panelidir: 10 yıllık risk havuzu, "
        "optimize senaryo kazancı, temel durumla kıyaslayan grafik ve InsurTech simülatörü. "
        "Üstteki sabit finansal değerler örnek veridir, canlı taramadan hesaplanmaz.")),
    "col_tiers": ("SaaS paketleri neler?", lambda c: (
        "<b>Basic/Standart</b> (sabit abonelik, küçük operatörler), <b>Pro/Enterprise</b> (kullanıma "
        "dayalı API, ticari filolar) ve <b>Institutional/InsurTech</b> (yıllık sözleşme, savunma "
        "kurumları ve sigortacılar).")),
    # --- Re-entry ---
    "re_how": ("Simülasyon nasıl çalışıyor?", lambda c: (
        "Nesne 120 km'de 7,8 km/s ile başlar. Üstel atmosfer, balistik sürükleme ve Sutton-Graves ısı "
        "akısı modeliyle irtifa, hız, sıcaklık ve yavaşlama 0,05 sn adımlarla hesaplanır. "
        "▶ TEKRAR OYNAT animasyonu yeniden oynatır.")),
    "re_demise": ("Parçalanma neye bağlı?", lambda c: (
        "Biriken ısı (MJ/kg) malzemenin eşiğine ulaşınca nesne parçalanır (<b>PARÇALANMA</b>): alüminyum 0,97, "
        "çelik 1,0, titanyum 1,15, karbon kompozit 3,0 MJ/kg.")),
    "re_params": ("Hangi ayarlar sonucu değiştirir?", lambda c: (
        "Dört şey: <b>kütle</b>, <b>kesit alanı</b>, 120 km'deki <b>uçuş yolu açısı</b> ve <b>malzeme</b>. "
        "Dik açı daha kısa ve sert, sığ açı daha uzun bir giriş demektir.")),
    "re_phase": ("Animasyon aşamaları?", lambda c: (
        "<b>ALÇALMA</b> alçalma, <b>🔥 PLAZMA ISINMASI</b> ısı akısı zirvenin %10'unu aşınca, "
        "<b>💥 PARÇALANMA</b> parçalanma anıdır.")),
    "re_carbon": ("Karbon kompozit neden farklı?", lambda c: (
        "Parçalanması için gereken ısı 3,0 MJ/kg ile alüminyumun yaklaşık 3 katıdır; yani daha zor "
        "yanar ve yere ulaşan parça olasılığı artabilir.")),
}

BOT_QA_EN = {
    # --- General ---
    "site": ("What does this site do?", lambda c: (
        "<b>SPACE GRAVE</b> is a space debris command center built for NASA Space Apps. "
        "It computes real orbit data (TLE) from CelesTrak with SGP4. "
        f"Right now <b>{c['n_active']:,}</b> active satellites and <b>{c['n_debris']:,}</b> debris fragments are tracked.")),
    "pages": ("Which pages are there?", lambda c: (
        "There are four modules: <b>General Status Overview</b> (summary cards and chart), "
        "<b>3D Earth &amp; Orbital Radar</b> (live 3D globe), <b>Collision Risk Analysis</b> "
        "(conjunction screening + OrbitRisk financial panel) and <b>Re-Entry &amp; Burn-Up Sim</b> "
        "(atmospheric re-entry simulation). Use the buttons at the top to switch.")),
    "veri": ("Where does the data come from?", lambda c: (
        "Data is pulled from <b>CelesTrak</b>: active satellites, visible satellites and the Cosmos 2251, "
        "Iridium 33, Fengyun-1C and Cosmos 1408 debris groups. Data is cached for 2 hours. "
        f"Last update: <b>{c['fetched']}</b>. Without a connection, fallback data for ISS, Hubble and Tiangong is used.")),
    "feed": ("What does the feed status mean?", lambda c: (
        "<b>LIVE 🟢</b> means live data, <b>CACHED 🟡</b> cache, <b>PARTIAL 🟡</b> some groups could not be fetched, "
        f"<b>OFFLINE 🟠</b> fallback data. Right now: <b>{c['feed']}</b>.")),
    "sgp4": ("What is SGP4?", lambda c: (
        "<b>SGP4</b> is the standard orbit model that computes an object's position and velocity at a given moment "
        "from TLE lines. The radar, collision screening and table values are all produced with it.")),
    "kessler": ("What is Kessler Syndrome?", lambda c: (
        "<b>Kessler Syndrome</b> is the chain reaction in which, as debris density in orbit grows, collisions "
        "create new fragments that cause even more collisions. The site makes this risk visible with collision "
        "screening and debris charts.")),
    # --- Overview ---
    "cards": ("What do the cards mean?", lambda c: (
        "<b>Catalog Objects</b> is active + debris total, <b>Debris Fragments</b> the debris count, "
        "<b>Active Sats</b> the active satellite count. <b>Close Approaches</b> fills in once you run a screening "
        f"on the Collision page (right now: {_bot_conj()}). <b>Defense Matrix</b> is currently <b>{_bot_defense()}</b>.")),
    "shells": ("What does the orbit chart show?", lambda c: (
        "The <b>altitude × inclination</b> distribution of objects: blue dots are active satellites, red ones debris. "
        "Dots clustering in the same place share the same orbital shell. For readability, a random "
        "sample below 2200 km is shown.")),
    "table": ("Explain the live table", lambda c: (
        "The table lists satellites from the 'visual' group; altitude, inclination, period and speed are "
        "computed live with SGP4. ISS, CSS and Tiangong are marked <b>CREWED</b>.")),
    "decaying": ("What does DECAYING mean?", lambda c: (
        "<b>DECAYING ⚠️</b> is the label for objects whose altitude drops below 300 km. "
        "At that height atmospheric drag lowers the orbit quickly.")),
    # --- 3D Radar ---
    "radar_use": ("How do I use the radar?", lambda c: (
        "Drag the globe to rotate it and use the wheel to zoom. Hover over an object to see its orbit "
        "line and its name, NORAD ID, altitude and velocity in the HUD on the right. The selector at the "
        "top left changes the time warp (60×–900×).")),
    "radar_colors": ("What do the colors mean?", lambda c: (
        "Small satellites with light grey bodies are active satellites, the <b>green</b> one is the ISS and the "
        "<b>red</b> pieces are debris. An object with a red ring around it came within 25 km of another "
        "in the collision screening (the HUD shows FLAGGED).")),
    "radar_calc": ("How are positions computed?", lambda c: (
        "For each object a 100-step orbit (60 s apart) is computed from the real TLE with SGP4; "
        "the browser interpolates between steps for smooth motion. The motion loops back about every 99 minutes.")),
    "radar_limit": ("Why aren't all objects shown?", lambda c: (
        "For performance, the sliders show at most 400 active satellites and 600 debris fragments (randomly "
        "picked). Satellites in the 'visual' group are always added.")),
    # --- Collision ---
    "col_scan": ("How does the screening work?", lambda c: (
        "Two stages: first, pairs below the threshold distance are found with a KD-Tree at 30 s steps; then, "
        "for the 40 closest pairs, the time of closest approach (TCA) is computed in a ±35 s window with 0.25 s steps. "
        "Objects flying together are filtered out.")),
    "col_risk": ("What are the risk levels?", lambda c: (
        "By miss distance: <b>&lt; 1 km</b> 🔴 CRITICAL, <b>&lt; 5 km</b> 🟠 HIGH, "
        "<b>&lt; 25 km</b> 🟡 WATCH, anything farther ⚪ SCREENED.")),
    "col_def": ("What is the Defense Matrix?", lambda c: (
        "If at least one approach below 5 km is found, the <b>Defense Matrix</b> becomes ARMED (shown on the "
        f"Overview card). Right now: <b>{_bot_defense()}</b>.")),
    "col_tca": ("What are miss distance & TCA?", lambda c: (
        "<b>Miss Distance</b> is how far apart two objects are at closest approach, <b>TCA</b> is the UTC time of that pass, "
        "and <b>Rel. Velocity</b> is their relative speed. The higher the relative velocity, the bigger the collision energy.")),
    "col_red": ("What does Red Alarm do?", lambda c: (
        "In the OrbitRisk panel at the bottom of the page, <b>🚨 Simulate Red Alarm</b> starts a demo scenario: "
        "a critical 11.2 km approach for Sat-Omega is added to the table and the 50 km safety sphere "
        "status switches to RED ALARM.")),
    "col_policy": ("How is the policy calculated?", lambda c: (
        "<b>Risk Score</b> = min(98, satellite count/10 × 0.4 + rocket factor × 20 + density factor × 15). "
        "<b>Annual policy risk</b> = satellite count × 18,500 × rocket factor × density factor "
        "(shown in millions of dollars).")),
    "col_fin": ("What is the OrbitRisk panel?", lambda c: (
        "The lower part of the Collision page is the <b>OrbitRisk</b> financial panel: 10-year risk pool, "
        "optimized-scenario savings, a chart comparing against the baseline and the InsurTech simulator. "
        "The fixed financial values at the top are sample data, not computed from the live screening.")),
    "col_tiers": ("What are the SaaS tiers?", lambda c: (
        "<b>Basic/Standard</b> (flat subscription, small operators), <b>Pro/Enterprise</b> (usage-based "
        "API, commercial fleets) and <b>Institutional/InsurTech</b> (annual contract, defense "
        "agencies and insurers).")),
    # --- Re-entry ---
    "re_how": ("How does the simulation work?", lambda c: (
        "The object starts at 120 km at 7.8 km/s. Altitude, velocity, temperature and deceleration are computed "
        "in 0.05 s steps with an exponential atmosphere, ballistic drag and a Sutton-Graves heat flux model. "
        "The ▶ REPLAY button plays the animation again.")),
    "re_demise": ("What does disintegration depend on?", lambda c: (
        "When accumulated heat (MJ/kg) reaches the material's threshold, the object breaks up (<b>DEMISE</b>): aluminum 0.97, "
        "steel 1.0, titanium 1.15, carbon composite 3.0 MJ/kg.")),
    "re_params": ("Which settings change the result?", lambda c: (
        "Four things: <b>mass</b>, <b>cross-section area</b>, the <b>flight-path angle</b> at 120 km and the <b>material</b>. "
        "A steep angle means a shorter, harsher entry, a shallow angle a longer one.")),
    "re_phase": ("What are the animation phases?", lambda c: (
        "<b>DESCENT</b> is the fall, <b>🔥 PLASMA HEATING</b> starts when heat flux exceeds 10% of its peak, "
        "and <b>💥 DEMISE</b> is the moment of break-up.")),
    "re_carbon": ("Why is carbon composite different?", lambda c: (
        "The heat needed to break it up is 3.0 MJ/kg, about 3 times that of aluminum; so it burns up "
        "less easily and the chance of fragments reaching the ground can increase.")),
}


def _bot_qa():
    return BOT_QA_TR if st.session_state.lang == "tr" else BOT_QA_EN


# Sayfa indeksine göre öne çıkan sorular (+ her sayfada 2 genel soru)
BOT_FEATURED = {
    0: ["cards", "shells", "table", "decaying"],
    1: ["radar_use", "radar_colors", "radar_calc", "radar_limit"],
    2: ["col_scan", "col_risk", "col_red", "col_policy"],
    3: ["re_how", "re_demise", "re_params", "re_phase"],
}
BOT_GENERAL = ["site", "veri"]


def bot_toggle():
    st.session_state.bot_open = not st.session_state.bot_open


def bot_ask(qid):
    log = st.session_state.bot_log
    log.append({"r": "user", "q": qid})
    log.append({"r": "bot", "q": qid})
    del log[:-30]


def bot_pick():
    choice = st.session_state.sg_bot_pick
    for qid, (label, _) in _bot_qa().items():
        if label == choice:
            bot_ask(qid)
            break
    st.session_state.sg_bot_pick = _bot_ph()


def _bot_bubbles(log):
    # column-reverse kullanıldığı için en yeni mesaj önce yazılır (kutu hep en altta açılır)
    # Mesajlar seçili dile göre gösterim anında üretilir -> dil değişince geçmiş de çevrilir
    qa, ctx, out = _bot_qa(), st.session_state.bot_ctx, []
    for m in reversed(log):
        if m.get("g"):
            txt = T("Selam! Ben <b>SG-Bot</b> 🤖 Bu sayfayla ilgili merak ettiklerini aşağıdaki sorulara dokunarak sorabilirsin.",
                    "Hi! I'm <b>SG-Bot</b> 🤖 Tap the questions below to ask anything about this page.")
        elif m["q"] not in qa:
            continue
        elif m["r"] == "user":
            txt = html_escape(qa[m["q"]][0])
        else:
            txt = qa[m["q"]][1](ctx)
        txt = txt.replace("$", "&#36;")  # $...$ LaTeX gibi yorumlanmasın
        out.append(f'<div class="sg-msg sg-{"user" if m["r"] == "user" else "bot"}">{txt}</div>')
    return "".join(out)


def _bot_container(key):
    # st.container(key=...) eski Streamlit sürümlerinde yok -> sayfa altında normal kutu olarak çalışır
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


st.session_state.bot_ctx = {
    "n_active": len(CAT["groups"]["active"]),
    "n_debris": len(CAT["debris"]),
    "feed": FEED_LABEL,
    "fetched": CAT["fetched"],
    "page_i": PAGES.index(st.session_state.active_page) if st.session_state.active_page in PAGES else 0,
}

if st.session_state.bot_open:
    render_lang_switch("calc(min(340px, 92vw) + 36px)", "22px")
else:
    render_lang_switch("96px", "31px")

st.markdown("""
    <style>
    .st-key-sg_bot_fab { position: fixed; bottom: 22px; right: 22px; width: 60px; z-index: 999999; }
    .st-key-sg_bot_fab button {
        width: 60px; height: 60px; border-radius: 50%; padding: 0; font-size: 26px;
        background: linear-gradient(135deg, #0b4b94 0%, #0a2a5c 100%);
        border: 1px solid #38bdf8; box-shadow: 0 0 25px rgba(56,189,248,0.6);
    }
    .st-key-sg_bot_win {
        position: fixed; bottom: 22px; right: 22px; width: min(340px, 92vw); max-height: 88vh;
        overflow-y: auto; z-index: 999999; gap: 0.45rem !important; padding: 12px; border-radius: 16px;
        background: linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(7,11,25,0.99) 100%);
        border: 1px solid rgba(0,102,204,0.8);
        box-shadow: 0 12px 40px rgba(0,0,0,0.85), 0 0 25px rgba(0,102,204,0.35);
    }
    .st-key-sg_bot_win [data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
    .st-key-sg_bot_win button {
        background: #0f1b33; border: 1px solid #1e3a5f; border-radius: 10px;
        padding: 4px 8px; min-height: 34px; line-height: 1.2;
    }
    .st-key-sg_bot_win button p { font-size: 11.5px; color: #cbd5e1; margin: 0; }
    .st-key-sg_bot_win button:hover { border-color: #38bdf8; }
    .st-key-sg_bot_win [data-baseweb="select"] > div { background: #0b1329; border-color: #1e3a5f; }
    .st-key-sg_bot_win [data-baseweb="select"] * { color: #e2e8f0 !important; font-size: 12px; }
    .sg-title { color: #38bdf8; font-weight: 800; font-size: 13px; letter-spacing: 1px; padding-top: 6px; }
    .sg-log {
        display: flex; flex-direction: column-reverse; gap: 6px; height: 210px; overflow-y: auto;
        padding: 6px; background: rgba(2,6,23,0.6); border: 1px solid #12315a; border-radius: 10px;
    }
    .sg-msg { font-size: 12px; line-height: 1.45; padding: 7px 10px; border-radius: 10px; max-width: 90%; color: #e2e8f0; }
    .sg-bot { align-self: flex-start; background: #0e2433; border-left: 3px solid #38bdf8; }
    .sg-user { align-self: flex-end; background: #1e3a8a; }
    </style>
""", unsafe_allow_html=True)

if st.session_state.bot_open:
    with _bot_container("sg_bot_win"):
        hc1, hc2 = st.columns([5, 1])
        hc1.markdown(f'<div class="sg-title">🛰️ SG-BOT · {T("HIZLI YARDIM", "QUICK HELP")}</div>', unsafe_allow_html=True)
        hc2.button("✕", key="sg_bot_close", on_click=bot_toggle)
        st.markdown(f'<div class="sg-log">{_bot_bubbles(st.session_state.bot_log)}</div>', unsafe_allow_html=True)
        qcols = st.columns(2)
        for i, qid in enumerate(BOT_FEATURED.get(st.session_state.bot_ctx["page_i"], []) + BOT_GENERAL):
            qcols[i % 2].button(_bot_qa()[qid][0], key=f"sg_q_{qid}", on_click=bot_ask, args=(qid,), **stretch(st.button))
        st.selectbox(T("Tüm sorular", "All questions"), [_bot_ph()] + [v[0] for v in _bot_qa().values()],
                     key="sg_bot_pick", on_change=bot_pick, label_visibility="collapsed")
else:
    with _bot_container("sg_bot_fab"):
        st.button("💬", key="sg_bot_open", on_click=bot_toggle)