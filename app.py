# -*- coding: utf-8 -*-
# ============================================================
# Painel Mesa de Análise — VELOX (multi-meses)
# ============================================================

import os, json, re, unicodedata
from datetime import datetime, date
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ------------------ CONFIG BÁSICA ------------------
st.set_page_config(page_title="Painel Mesa de Análise — VELOX", layout="wide")
st.title("🧮 Painel Mesa de Análise — VELOX")

# Cor padrão dos gráficos
CHART_COLOR = "#730000"

st.markdown(
    """
<style>
.card-wrap{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 6px;}
.card{background:#f7f7f9;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06);padding:14px 16px;min-width:220px;flex:1;text-align:center}
.card h4{margin:0 0 6px;font-size:14px;color:#004a7c;font-weight:700}
.card h2{margin:0;font-size:26px;font-weight:800;color:#222}
.card .sub{margin-top:8px;display:inline-block;padding:6px 10px;border-radius:8px;font-size:12px;font-weight:700}
.sub.neu{background:#f1f1f4;color:#444;border:1px solid #e4e4e8}
.section{font-size:18px;font-weight:800;margin:22px 0 8px}
.small{color:#666;font-size:13px}
.table-note{margin-top:8px;color:#666;font-size:12px}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-weight:800;font-size:12px}
.badge.ok{background:#e7f6ea;color:#1b5e20;border:1px solid #bfe6c7}
.badge.low{background:#fdecea;color:#8a1c1c;border:1px solid #f5c2c2}
.badge.high{background:#fff4e5;color:#7a4a00;border:1px solid #ffd59a}
</style>
""",
    unsafe_allow_html=True,
)

fast_mode = st.toggle("Modo rápido (pular alguns gráficos pesados)", value=False)

# ------------------ HELPERS GERAIS ------------------
ID_RE = re.compile(r"/d/([a-zA-Z0-9-_]+)")

def _sheet_id(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = ID_RE.search(s)
    if m:
        return m.group(1)
    return s if re.fullmatch(r"[A-Za-z0-9-_]{20,}", s) else None

def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(ch)
    )

def _upper(x):
    return str(x).upper().strip() if pd.notna(x) else ""

def parse_date_any(x):
    if pd.isna(x) or x == "":
        return pd.NaT
    # excel serial
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return (pd.to_datetime("1899-12-30") +
                    pd.to_timedelta(int(x), unit="D")).date()
        except Exception:
            pass
    s = str(x).strip()
    # tenta datetime com hora
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # tenta só data
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return pd.NaT

def parse_time_seconds(x) -> int:
    """Converte HH:MM:SS ou MM:SS em segundos."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    s = str(x).strip()
    if not s:
        return 0
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except Exception:
        return 0
    if len(parts) == 3:
        h, m, s2 = parts
    elif len(parts) == 2:
        h, m, s2 = 0, parts[0], parts[1]
    else:
        return 0
    return h * 3600 + m * 60 + s2

def format_seconds_mmss(sec: float) -> str:
    """Mostra em minutos (MM:SS) ou HH:MM:SS se passar de 1h."""
    if sec is None or np.isnan(sec) or sec <= 0:
        return "—"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def bar_with_labels(df, x_col, y_col, x_title="", y_title="QTD", height=320):
    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x_col}:N",
            sort="-y",
            title=x_title,
            axis=alt.Axis(labelAngle=0, labelLimit=180, labelOverlap=False),
        ),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[x_col, y_col],
    )
    bars = base.mark_bar(color=CHART_COLOR)
    labels = base.mark_text(dy=-6).encode(
        text=alt.Text(f"{y_col}:Q", format=".0f")
    )
    return (bars + labels).properties(height=height)

def bar_with_qty_and_perc(df, x_col, y_qty, y_perc, x_title="", height=320):
    """
    Rótulo: "QTD (xx,x%)"
    Eixo Y = QTD, mas mostra % no rótulo.
    """
    df = df.copy()
    df["LABEL"] = df.apply(
        lambda r: f"{int(r[y_qty])} ({float(r[y_perc]):.1f}%)".replace(".", ","), axis=1
    )

    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x_col}:N",
            sort="-y",
            title=x_title,
            axis=alt.Axis(labelAngle=0, labelLimit=180, labelOverlap=False),
        ),
        y=alt.Y(f"{y_qty}:Q", title="Quantidade de vistorias"),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title=x_title or x_col),
            alt.Tooltip(f"{y_qty}:Q", title="Quantidade", format=".0f"),
            alt.Tooltip(f"{y_perc}:Q", title="%", format=".1f"),
        ],
    )
    bars = base.mark_bar(color=CHART_COLOR)
    labels = base.mark_text(dy=-6).encode(text="LABEL:N")
    return (bars + labels).properties(height=height)

def looks_like_plate(s: str) -> bool:
    """
    Heurística simples: placa Mercosul/antiga costuma ter 7 caracteres e mistura letras/números.
    Se não parecer placa, tratamos como chassi (1º emplacamento).
    """
    if s is None:
        return False
    t = str(s).strip().upper()
    if not t:
        return False
    t = re.sub(r"[^A-Z0-9]", "", t)
    return bool(re.fullmatch(r"[A-Z]{3}\d[A-Z]\d{2}", t) or re.fullmatch(r"[A-Z]{3}\d{4}", t))

# ------------------ CONEXÃO COM GOOGLE ------------------
def _get_client():
    try:
        block = st.secrets["gcp_service_account"]
    except Exception:
        st.error("Não encontrei [gcp_service_account] no .streamlit/secrets.toml.")
        st.stop()

    if "json_path" in block:
        path = block["json_path"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception as e:
            st.error(f"Não consegui abrir o JSON da service account: {path}")
            with st.expander("Detalhes"):
                st.exception(e)
            st.stop()
    else:
        info = dict(block)

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scopes)
    gc = gspread.authorize(creds)
    return gc

client = _get_client()

# ------------------ LEITURA DO ÍNDICE ------------------
ANALISTAS_INDEX_ID = st.secrets.get("analistas_index_sheet_id", "").strip()
if not ANALISTAS_INDEX_ID:
    st.error("Faltou `analistas_index_sheet_id` no secrets.toml")
    st.stop()

@st.cache_data(ttl=300, show_spinner=False)
def read_index(sheet_id: str, tab: str) -> pd.DataFrame:
    sh = client.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab)
    except Exception as e:
        raise RuntimeError(f"O índice não possui aba '{tab}'.") from e
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(columns=["URL", "MÊS", "ATIVO"])
    df = pd.DataFrame(rows)
    df.columns = [c.strip().upper() for c in df.columns]
    for need in ["URL", "MÊS", "ATIVO"]:
        if need not in df.columns:
            df[need] = ""
    return df

# ------------------ LEITURA MENSAL — PRODUÇÃO ------------------
@st.cache_data(ttl=300, show_spinner=False)
def read_producao_month(sheet_id: str) -> Tuple[pd.DataFrame, str]:
    sh = client.open_by_key(sheet_id)
    title = sh.title or sheet_id
    ws = sh.sheet1
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(), title

    df = pd.DataFrame(rows)
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        cu = _strip_accents(c).upper()
        if "ORDEM" in cu and "SERVICO" in cu:
            rename[c] = "OS"
        elif cu == "PLACA":
            rename[c] = "PLACA"
        elif "DATA/HORA V6".replace("/", "") in cu.replace(" ", "") or "DATAHORA" in cu:
            rename[c] = "DATA_HORA_V6"
        elif "DATA ABERTURA" in cu and "MESA" in cu:
            rename[c] = "DATA_ABERTURA_MESA"
        elif "HORA ABERTURA" in cu and "MESA" in cu:
            rename[c] = "HORA_ABERTURA_MESA"
        elif "STATUS" in cu and "LAUDO" in cu:
            rename[c] = "STATUS_LAUDO"
        elif "TIPO" in cu and "USUARIO" in cu:
            rename[c] = "TIPO_USUARIO"
        elif cu == "USUARIO" or "USUARIO" in cu:
            rename[c] = "USUARIO"
        elif "TEMPO" in cu and "TOTAL" in cu:
            rename[c] = "TEMPO_TOTAL"

    df = df.rename(columns=rename)

    for need in [
        "OS","PLACA","DATA_HORA_V6","DATA_ABERTURA_MESA","HORA_ABERTURA_MESA",
        "STATUS_LAUDO","TIPO_USUARIO","USUARIO","TEMPO_TOTAL"
    ]:
        if need not in df.columns:
            df[need] = ""

    if df["DATA_ABERTURA_MESA"].astype(str).str.strip().ne("").any():
        df["DATA_BASE"] = df["DATA_ABERTURA_MESA"].apply(parse_date_any)
    else:
        df["DATA_BASE"] = df["DATA_HORA_V6"].apply(parse_date_any)

    df["OS"] = df["OS"].astype(str).str.strip()
    df["PLACA"] = df["PLACA"].astype(str).str.strip()
    df["TIPO_USUARIO"] = df["TIPO_USUARIO"].astype(str).map(_upper)
    df["USUARIO"] = df["USUARIO"].astype(str).map(_upper)
    df["STATUS_LAUDO"] = df["STATUS_LAUDO"].astype(str).str.strip()
    df["TEMPO_SEG"] = df["TEMPO_TOTAL"].apply(parse_time_seconds)

    return df, title

# ------------------ LEITURA MENSAL — CRÍTICA ------------------
@st.cache_data(ttl=300, show_spinner=False)
def read_critica_month(sheet_id: str) -> Tuple[pd.DataFrame, str]:
    sh = client.open_by_key(sheet_id)
    title = sh.title or sheet_id
    ws = sh.sheet1
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(), title

    df = pd.DataFrame(rows)
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        cu = _strip_accents(c).upper()
        if "ORDEM" in cu and "SERVICO" in cu:
            rename[c] = "OS"
        elif cu == "PLACA":
            rename[c] = "PLACA"
        elif "VISTORIADOR" in cu:
            rename[c] = "VISTORIADOR"
        elif "ANALISTA" in cu:
            rename[c] = "ANALISTA"
        elif "STATUS" in cu and "CRITICA" in cu:
            rename[c] = "STATUS_CRITICA"
        elif "DATA" in cu and "CRITICA" in cu:
            rename[c] = "DATA_CRITICA"
        elif "OBSERVACAO" in cu or "OBSERVAÇÃO" in cu or cu.startswith("OBS"):
            rename[c] = "OBS"

    df = df.rename(columns=rename)

    for need in ["OS","PLACA","VISTORIADOR","ANALISTA","STATUS_CRITICA","DATA_CRITICA","OBS"]:
        if need not in df.columns:
            df[need] = ""

    df["DATA_CRITICA"] = df["DATA_CRITICA"].apply(parse_date_any)
    df["OS"] = df["OS"].astype(str).str.strip()
    df["PLACA"] = df["PLACA"].astype(str).map(_upper)
    df["VISTORIADOR"] = df["VISTORIADOR"].astype(str).map(_upper)
    df["ANALISTA"] = df["ANALISTA"].astype(str).map(_upper)
    df["STATUS_CRITICA"] = df["STATUS_CRITICA"].astype(str).map(_upper)

    return df, title

# ------------------ CARREGA ÍNDICES ------------------
idx_prod = read_index(ANALISTAS_INDEX_ID, tab="PRODUÇÃO")
idx_crit = read_index(ANALISTAS_INDEX_ID, tab="CRÍTICA")

if "ATIVO" in idx_prod.columns:
    idx_prod["ATIVO"] = idx_prod["ATIVO"].astype(str).map(_upper)
    idx_prod = idx_prod[idx_prod["ATIVO"].isin(["S","SIM","1","Y","YES"])]
if "ATIVO" in idx_crit.columns:
    idx_crit["ATIVO"] = idx_crit["ATIVO"].astype(str).map(_upper)
    idx_crit = idx_crit[idx_crit["ATIVO"].isin(["S","SIM","1","Y","YES"])]

sel_meses = sorted([str(m).strip() for m in idx_prod["MÊS"] if str(m).strip()])
if not sel_meses:
    st.error("Índice de PRODUÇÃO sem meses ativos.")
    st.stop()

idx_prod = idx_prod[idx_prod["MÊS"].isin(sel_meses)]
idx_crit = idx_crit[idx_crit["MÊS"].isin(sel_meses)]

# ------------------ LÊ TODOS OS MESES ------------------
prod_all, crit_all = [], []

for _, r in idx_prod.iterrows():
    sid = _sheet_id(r["URL"])
    if not sid:
        continue
    df, _ = read_producao_month(sid)
    if not df.empty:
        prod_all.append(df)

for _, r in idx_crit.iterrows():
    sid = _sheet_id(r["URL"])
    if not sid:
        continue
    df, _ = read_critica_month(sid)
    if not df.empty:
        crit_all.append(df)

if not prod_all:
    st.error("Não consegui ler dados de PRODUÇÃO de nenhum mês.")
    st.stop()

dfProd = pd.concat(prod_all, ignore_index=True)
dfCrit = (
    pd.concat(crit_all, ignore_index=True)
    if crit_all
    else pd.DataFrame(columns=["OS","PLACA","VISTORIADOR","ANALISTA","STATUS_CRITICA","DATA_CRITICA","OBS"])
)

# ------------------ FILTRO DE MÊS E PERÍODO ------------------
s_all_dt = pd.to_datetime(dfProd["DATA_BASE"], errors="coerce")
ym_all = s_all_dt.dt.to_period("M").dropna().astype(str).unique().tolist()
ym_all = sorted(ym_all)
if not ym_all:
    st.error("Base de produção sem datas válidas.")
    st.stop()

label_map = {f"{m[5:]}/{m[:4]}": m for m in ym_all}
sel_label = st.selectbox("Mês de referência", options=list(label_map.keys()), index=len(ym_all) - 1)
ym_sel = label_map[sel_label]
ref_year, ref_month = int(ym_sel[:4]), int(ym_sel[5:7])

mask_mes = s_all_dt.dt.year.eq(ref_year) & s_all_dt.dt.month.eq(ref_month)
dfProd_mes = dfProd[mask_mes].copy()

s_mes_dates = pd.to_datetime(dfProd_mes["DATA_BASE"], errors="coerce").dt.date
min_d, max_d = min(s_mes_dates.dropna()), max(s_mes_dates.dropna())

col1, col2 = st.columns([1.2, 2.8])
with col1:
    drange = st.date_input(
        "Período (dentro do mês)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        format="DD/MM/YYYY",
    )

start_d, end_d = drange if isinstance(drange, tuple) and len(drange) == 2 else (min_d, max_d)

mask_dias = s_mes_dates.map(lambda d: isinstance(d, date) and start_d <= d <= end_d)
viewProd = dfProd_mes[mask_dias].copy()

# recorte correspondente em CRÍTICA
if not dfCrit.empty:
    s_crit = pd.to_datetime(dfCrit["DATA_CRITICA"], errors="coerce")
    mask_crit_mes = s_crit.dt.year.eq(ref_year) & s_crit.dt.month.eq(ref_month)
    dfCrit_mes = dfCrit[mask_crit_mes].copy()

    s_crit_dates = pd.to_datetime(dfCrit_mes["DATA_CRITICA"], errors="coerce").dt.date
    mask_crit_dias = s_crit_dates.map(lambda d: isinstance(d, date) and start_d <= d <= end_d)
    viewCrit = dfCrit_mes[mask_crit_dias].copy()
else:
    dfCrit_mes = pd.DataFrame()
    viewCrit = pd.DataFrame()

# -------- Filtros adicionais (analista / vistoriador) --------
df_analistas = viewProd[viewProd["TIPO_USUARIO"] == "ANALISTA MESA"]
analistas_opts = sorted(df_analistas["USUARIO"].dropna().unique().tolist())

df_vist = viewProd[viewProd["TIPO_USUARIO"] == "VISTORIADOR"]
vist_opts = sorted(df_vist["USUARIO"].dropna().unique().tolist())

with col2:
    c21, c22 = st.columns(2)
    with c21:
        f_analistas = st.multiselect("Analistas (opcional)", analistas_opts)
    with c22:
        f_vists = st.multiselect("Vistoriadores (opcional)", vist_opts)

if f_analistas:
    ups = [_upper(a) for a in f_analistas]
    mask_keep = (viewProd["TIPO_USUARIO"] != "ANALISTA MESA") | (viewProd["USUARIO"].isin(ups))
    viewProd = viewProd[mask_keep]
    if not viewCrit.empty and "ANALISTA" in viewCrit.columns:
        viewCrit = viewCrit[viewCrit["ANALISTA"].isin(ups)]

if f_vists:
    ups = [_upper(v) for v in f_vists]
    mask_keep = (viewProd["TIPO_USUARIO"] != "VISTORIADOR") | (viewProd["USUARIO"].isin(ups))
    viewProd = viewProd[mask_keep]
    if not viewCrit.empty and "VISTORIADOR" in viewCrit.columns:
        viewCrit = viewCrit[viewCrit["VISTORIADOR"].isin(ups)]

if viewProd.empty:
    st.info("Sem registros de produção da mesa de análise no período/filtros.")
    st.stop()

# ------------------ KPIs PRINCIPAIS ------------------
base_analista = viewProd[viewProd["TIPO_USUARIO"] == "ANALISTA MESA"].copy()
base_fila = viewProd[viewProd["TIPO_USUARIO"] == "FILA MESA"].copy()
base_vist = viewProd[viewProd["TIPO_USUARIO"] == "VISTORIADOR"].copy()

total_registros_analista = int(len(base_analista))
analistas_avaliados = int(base_analista["USUARIO"].nunique()) if not base_analista.empty else 0

tempo_medio_analista = base_analista["TEMPO_SEG"].mean() if not base_analista.empty else np.nan
tempo_medio_fila = base_fila["TEMPO_SEG"].mean() if not base_fila.empty else np.nan
tempo_medio_vist = base_vist["TEMPO_SEG"].mean() if not base_vist.empty else np.nan

tempo_total_por_os = viewProd.groupby("OS")["TEMPO_SEG"].sum()
tempo_medio_total_proc = tempo_total_por_os.mean() if len(tempo_total_por_os) else np.nan

cards_html = f"""
<div class="card-wrap">
  <div class='card'>
    <h4>Registros de análise (Analista Mesa)</h4>
    <h2>{total_registros_analista:,}</h2>
  </div>
  <div class='card'>
    <h4>Analistas avaliados</h4>
    <h2>{analistas_avaliados:,}</h2>
  </div>
  <div class='card'>
    <h4>Tempo médio de análise</h4>
    <h2>{format_seconds_mmss(tempo_medio_analista)}</h2>
    <span class='sub neu'>Base: registros "Analista Mesa"</span>
  </div>
  <div class='card'>
    <h4>Tempo médio em fila da mesa</h4>
    <h2>{format_seconds_mmss(tempo_medio_fila)}</h2>
    <span class='sub neu'>Base: registros "Fila Mesa"</span>
  </div>
  <div class='card'>
    <h4>Tempo médio do vistoriador</h4>
    <h2>{format_seconds_mmss(tempo_medio_vist)}</h2>
    <span class='sub neu'>Registros "Vistoriador"</span>
  </div>
  <div class='card'>
    <h4>Tempo médio total por vistoria</h4>
    <h2>{format_seconds_mmss(tempo_medio_total_proc)}</h2>
    <span class='sub neu'>Soma de todas as etapas da OS</span>
  </div>
</div>
""".replace(",", ".")
st.markdown(cards_html, unsafe_allow_html=True)

# ------------------ GRÁFICOS PRINCIPAIS (analistas) ------------------
c1, c2 = st.columns(2)

if not base_analista.empty:
    with c1:
        st.markdown('<div class="section">Volume de análises por analista</div>', unsafe_allow_html=True)
        vol = (
            base_analista.groupby("USUARIO", dropna=False)["OS"]
            .size()
            .reset_index(name="QTD")
            .sort_values("QTD", ascending=False)
        )
        st.altair_chart(bar_with_labels(vol, "USUARIO", "QTD", x_title="ANALISTA"), use_container_width=True)

if not base_analista.empty:
    with c2:
        st.markdown('<div class="section">Tempo médio de análise por analista</div>', unsafe_allow_html=True)
        tmp = (
            base_analista.groupby("USUARIO", dropna=False)["TEMPO_SEG"]
            .mean()
            .reset_index(name="SEG_MEDIO")
            .sort_values("SEG_MEDIO", ascending=False)
        )
        tmp["TEMPO_MEDIO"] = tmp["SEG_MEDIO"].apply(format_seconds_mmss)

        base_chart = alt.Chart(tmp).encode(
            x=alt.X("USUARIO:N", axis=alt.Axis(labelAngle=0, labelLimit=180), title="ANALISTA"),
            y=alt.Y("SEG_MEDIO:Q", title="Tempo médio (segundos)"),
            tooltip=[
                "USUARIO:N",
                alt.Tooltip("TEMPO_MEDIO:N", title="Tempo (mm:ss)"),
                alt.Tooltip("SEG_MEDIO:Q", title="Segundos", format=".1f"),
            ],
        )
        bars = base_chart.mark_bar(color=CHART_COLOR)
        labels = base_chart.mark_text(dy=-6).encode(text=alt.Text("TEMPO_MEDIO:N"))
        st.altair_chart((bars + labels).properties(height=340), use_container_width=True)

# ------------------ NOVO: PARTICIPAÇÃO DA PRODUÇÃO POR ANALISTA (META 25% ±3pp) ------------------
st.markdown("---")
st.markdown('<div class="section">Produção por analista em % (meta: 25% ± 3pp)</div>', unsafe_allow_html=True)

META_PERC = 25.0
MARGEM_PP = 3.0
LIM_INF = META_PERC - MARGEM_PP  # 22
LIM_SUP = META_PERC + MARGEM_PP  # 28

if base_analista.empty:
    st.info("Sem registros de 'ANALISTA MESA' no recorte atual para calcular a participação.")
else:
    share = (
        base_analista.groupby("USUARIO", dropna=False)["OS"]
        .size()
        .reset_index(name="QTD")
    )
    total_qtd = int(share["QTD"].sum())
    share["PERC"] = share["QTD"] / total_qtd * 100 if total_qtd else 0.0
    share["DELTA_PP"] = share["PERC"] - META_PERC

    def _status_row(p):
        if p < LIM_INF:
            return "Abaixo"
        if p > LIM_SUP:
            return "Acima"
        return "Dentro"

    share["STATUS"] = share["PERC"].apply(_status_row)

    # Indicadores (cards por analista)
    share_sorted = share.sort_values("PERC", ascending=False).copy()
    cols = st.columns(min(len(share_sorted), 4))
    for i, (_, r) in enumerate(share_sorted.iterrows()):
        col = cols[i % len(cols)]
        perc_txt = f"{r['PERC']:.1f}%".replace(".", ",")
        delta_txt = f"{r['DELTA_PP']:+.1f} pp".replace(".", ",")
        if r["STATUS"] == "Dentro":
            badge = "<span class='badge ok'>DENTRO</span>"
            arrow = "▲" if r["DELTA_PP"] >= 0 else "▼"
        elif r["STATUS"] == "Abaixo":
            badge = "<span class='badge low'>ABAIXO</span>"
            arrow = "▼"
        else:
            badge = "<span class='badge high'>ACIMA</span>"
            arrow = "▲"

        with col:
            st.markdown(
                f"""
                <div class='card'>
                  <h4>{r['USUARIO']}</h4>
                  <h2>{perc_txt}</h2>
                  <div class='small'>{arrow} {delta_txt} vs 25%</div>
                  <div style="margin-top:8px">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Gráfico em % + linhas de meta e limites
    share_plot = share_sorted.copy()
    share_plot["PERC_LABEL"] = share_plot["PERC"].map(lambda v: f"{v:.1f}%".replace(".", ","))

    base_s = alt.Chart(share_plot).encode(
        x=alt.X("USUARIO:N", title="Analista", axis=alt.Axis(labelAngle=0, labelLimit=180)),
        y=alt.Y("PERC:Q", title="% da produção (Analista Mesa)"),
        tooltip=[
            alt.Tooltip("USUARIO:N", title="Analista"),
            alt.Tooltip("QTD:Q", title="Qtd análises", format=".0f"),
            alt.Tooltip("PERC:Q", title="%", format=".1f"),
            alt.Tooltip("STATUS:N", title="Status"),
        ],
    )

    bars_s = base_s.mark_bar(color=CHART_COLOR)
    labels_s = base_s.mark_text(dy=-6).encode(text="PERC_LABEL:N")

    # Linhas (meta e margem)
    df_rules = pd.DataFrame(
        {
            "VALOR": [LIM_INF, META_PERC, LIM_SUP],
            "TIPO": ["Limite inferior (22%)", "Meta (25%)", "Limite superior (28%)"],
        }
    )
    rules = alt.Chart(df_rules).mark_rule(strokeDash=[6, 4], color="#666").encode(
        y="VALOR:Q",
        tooltip=[alt.Tooltip("TIPO:N", title="Referência"), alt.Tooltip("VALOR:Q", title="%", format=".1f")],
    )

    st.altair_chart((bars_s + labels_s + rules).properties(height=320), use_container_width=True)

    # Tabela resumida (opcional, já com status)
    resumo = share_sorted[["USUARIO", "QTD", "PERC", "STATUS"]].copy()
    resumo["PERC"] = resumo["PERC"].map(lambda v: f"{v:.1f}%".replace(".", ","))
    st.dataframe(resumo, use_container_width=True, hide_index=True)

# ------------------ TEMPO TOTAL POR ETAPA (IMPORTANTE) ------------------
st.markdown("---")
st.markdown('<div class="section">Tempo total por etapa (mesa / fila / vistoriador)</div>', unsafe_allow_html=True)

tempo_tot = (
    viewProd.groupby("TIPO_USUARIO")["TEMPO_SEG"]
    .sum()
    .reset_index(name="TOTAL_SEG")
    .sort_values("TOTAL_SEG", ascending=False)
)
tempo_tot["TEMPO_TOTAL"] = tempo_tot["TOTAL_SEG"].apply(format_seconds_mmss)

base_tt = alt.Chart(tempo_tot).encode(
    x=alt.X("TIPO_USUARIO:N", title="Etapa", axis=alt.Axis(labelAngle=0, labelLimit=180)),
    y=alt.Y("TOTAL_SEG:Q", title="Tempo total (segundos)"),
    tooltip=[
        alt.Tooltip("TIPO_USUARIO:N", title="Etapa"),
        alt.Tooltip("TEMPO_TOTAL:N", title="Tempo total"),
        alt.Tooltip("TOTAL_SEG:Q", title="Segundos", format=".0f"),
    ],
)
bars_tt = base_tt.mark_bar(color=CHART_COLOR)
labels_tt = base_tt.mark_text(dy=-6).encode(text="TEMPO_TOTAL:N")
st.altair_chart((bars_tt + labels_tt).properties(height=320), use_container_width=True)

# ------------------ ÍNDICES DE QUALIDADE ------------------
st.markdown("---")
st.markdown('<div class="section">Índices de qualidade</div>', unsafe_allow_html=True)

q1, q2, q3 = st.columns(3)

# 1) ÍNDICE DE APONTAMENTOS (CRÍTICA, 1ª análise) — COM FALLBACK (NÃO ZERA)
with q1:
    st.subheader("Índice de apontamentos (mesa de análise)")

    if not viewCrit.empty:
        base_crit = viewCrit.copy()
    elif "dfCrit_mes" in locals() and not dfCrit_mes.empty:
        base_crit = dfCrit_mes.copy()
    elif not dfCrit.empty:
        base_crit = dfCrit.copy()
    else:
        base_crit = pd.DataFrame()

    if base_crit.empty:
        st.info("Sem registros na base de CRÍTICA para o recorte atual.")
    else:
        base_crit = base_crit.dropna(subset=["OS"]).copy()
        base_crit["STATUS_CRITICA"] = base_crit["STATUS_CRITICA"].astype(str).str.upper().str.strip()

        if "DATA_CRITICA" in base_crit.columns:
            base_crit = base_crit.sort_values("DATA_CRITICA")

        primeiras = base_crit.groupby("OS", as_index=False).first()

        total = int(len(primeiras))
        reprov = int(primeiras["STATUS_CRITICA"].str.contains("REPROV", na=False).sum())
        aprov = int(total - reprov)

        df_apont = pd.DataFrame(
            {
                "CATEGORIA": ["Aprovada de primeira", "Teve apontamento (não aprovada de 1ª)"],
                "QTD": [aprov, reprov],
            }
        )
        df_apont["PERC"] = (df_apont["QTD"] / total * 100) if total else 0

        st.metric(
            "Vistorias com apontamento (1ª análise)",
            (f"{df_apont.loc[1, 'PERC']:.1f}%".replace(".", ",")) if total else "—",
        )

        st.altair_chart(
            bar_with_qty_and_perc(df_apont, "CATEGORIA", "QTD", "PERC", x_title="Situação na 1ª análise"),
            use_container_width=True,
        )

# 2) ÍNDICE DE APROVAÇÃO DO LAUDO (PRODUÇÃO)
with q2:
    st.subheader("Índice de aprovação do laudo (produção)")

    laudo_q = viewProd.copy()
    if "STATUS_LAUDO" not in laudo_q.columns:
        st.info("A base de produção não possui STATUS_LAUDO.")
    else:
        laudo_q["STATUS_LAUDO"] = laudo_q["STATUS_LAUDO"].astype(str).str.strip()
        laudo_q = laudo_q[laudo_q["STATUS_LAUDO"] != ""]

        if laudo_q.empty:
            st.info("Sem informações de STATUS_LAUDO para o recorte atual.")
        else:
            laudo_q = laudo_q.sort_values(["OS", "DATA_BASE"])
            last_laudo = laudo_q.dropna(subset=["OS"]).drop_duplicates("OS", keep="last")[["OS", "STATUS_LAUDO"]]

            total_os = int(last_laudo["OS"].nunique())
            status_counts = (
                last_laudo.groupby("STATUS_LAUDO")["OS"]
                .nunique()
                .reset_index(name="QTD")
                .sort_values("QTD", ascending=False)
            )
            status_counts["PERC"] = status_counts["QTD"] / total_os * 100

            def _cat_status(s: str) -> str:
                su = _strip_accents(s).upper()
                if su.startswith("APROVADO"):
                    return "Aprovadas"
                if su.startswith("REPROVADO"):
                    return "Reprovadas"
                return "Outros"

            agg = status_counts.copy()
            agg["CAT"] = agg["STATUS_LAUDO"].apply(_cat_status)
            agg_tot = agg.groupby("CAT")["QTD"].sum().reset_index(name="QTD")
            agg_tot["PERC"] = agg_tot["QTD"] / total_os * 100

            perc_aprov = float(agg_tot.loc[agg_tot["CAT"] == "Aprovadas", "PERC"].fillna(0))
            perc_repr = float(agg_tot.loc[agg_tot["CAT"] == "Reprovadas", "PERC"].fillna(0))

            st.markdown(
                f"<div class='small'><b>{perc_aprov:.1f}%</b> com laudo aprovado (vs <b>{perc_repr:.1f}%</b> reprovadas).</div>".replace(".", ","),
                unsafe_allow_html=True,
            )

            base_l = alt.Chart(status_counts).encode(
                x=alt.X("STATUS_LAUDO:N", title="Situação do laudo", axis=alt.Axis(labelAngle=0, labelLimit=180)),
                y=alt.Y("QTD:Q", title="Quantidade"),
                tooltip=[
                    alt.Tooltip("STATUS_LAUDO:N", title="Status"),
                    alt.Tooltip("QTD:Q", title="Qtd", format=".0f"),
                    alt.Tooltip("PERC:Q", title="%", format=".1f"),
                ],
            )
            bars_l = base_l.mark_bar(color=CHART_COLOR)
            labels_l = base_l.mark_text(dy=-6).encode(text=alt.Text("PERC:Q", format=".1f"))
            st.altair_chart((bars_l + labels_l).properties(height=320), use_container_width=True)

# 3) 1º EMPLACAMENTO vs TRANSFERÊNCIA
with q3:
    st.subheader("1º emplacamento vs transferência")

    tmp_os = viewProd.dropna(subset=["OS"]).copy()
    tmp_os = tmp_os.sort_values(["OS", "DATA_BASE"]).drop_duplicates("OS", keep="last")

    tmp_os["TIPO_VISTORIA"] = tmp_os["PLACA"].apply(lambda x: "Transferência" if looks_like_plate(x) else "1º emplacamento")
    dist = tmp_os.groupby("TIPO_VISTORIA")["OS"].nunique().reset_index(name="QTD")
    total_v = dist["QTD"].sum()
    dist["PERC"] = dist["QTD"] / total_v * 100 if total_v else 0

    base_v = alt.Chart(dist).encode(
        x=alt.X("TIPO_VISTORIA:N", title="Tipo de vistoria", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("QTD:Q", title="Quantidade"),
        tooltip=[
            alt.Tooltip("TIPO_VISTORIA:N", title="Tipo"),
            alt.Tooltip("QTD:Q", title="Qtd", format=".0f"),
            alt.Tooltip("PERC:Q", title="%", format=".1f"),
        ],
    )
    bars_v = base_v.mark_bar(color=CHART_COLOR)
    labels_v = base_v.mark_text(dy=-6).encode(text=alt.Text("PERC:Q", format=".1f"))
    st.altair_chart((bars_v + labels_v).properties(height=320), use_container_width=True)

    st.caption("Regra: se o campo PLACA parece placa (ABC1234 / ABC1D23) = Transferência; senão = 1º emplacamento.")

# ------------------ DETALHAMENTO (PLACAS / TABELAS) ------------------
if not fast_mode:
    st.markdown("---")
    st.markdown('<div class="section">Detalhamento por placa</div>', unsafe_allow_html=True)

    placas_opts = sorted(viewProd["PLACA"].dropna().astype(str).str.strip().unique().tolist()) if "PLACA" in viewProd.columns else []
    if placas_opts:
        sel_placa = st.selectbox("Escolha a placa", options=placas_opts)
        hist_prod = viewProd[viewProd["PLACA"].astype(str).str.strip() == sel_placa].copy()
        hist_prod["DATA_BASE"] = pd.to_datetime(hist_prod["DATA_BASE"], errors="coerce").dt.date
        hist_prod = hist_prod.sort_values(["DATA_BASE", "OS", "TIPO_USUARIO"])

        st.subheader("Histórico de produção da placa")
        cols = ["DATA_BASE","OS","PLACA","TIPO_USUARIO","USUARIO","STATUS_LAUDO","TEMPO_TOTAL"]
        for c in cols:
            if c not in hist_prod.columns:
                hist_prod[c] = ""
        hist_prod = hist_prod[cols].rename(columns={
            "DATA_BASE": "DATA",
            "TIPO_USUARIO": "TIPO USUÁRIO",
            "STATUS_LAUDO": "STATUS LAUDO",
            "TEMPO_TOTAL": "TEMPO TOTAL",
        })
        st.dataframe(hist_prod, use_container_width=True, hide_index=True)

        if not viewCrit.empty:
            hist_crit = viewCrit[viewCrit["PLACA"] == _upper(sel_placa)].copy()
            if not hist_crit.empty:
                st.subheader("Críticas relacionadas à placa")
                cols_c = ["DATA_CRITICA","OS","PLACA","VISTORIADOR","ANALISTA","STATUS_CRITICA","OBS"]
                for c in cols_c:
                    if c not in hist_crit.columns:
                        hist_crit[c] = ""
                hist_crit = hist_crit[cols_c].rename(columns={
                    "DATA_CRITICA": "DATA CRÍTICA",
                    "STATUS_CRITICA": "STATUS CRÍTICA",
                    "OBS": "OBSERVAÇÃO",
                })
                hist_crit = hist_crit.sort_values(["DATA CRÍTICA", "ANALISTA", "VISTORIADOR"])
                st.dataframe(hist_crit, use_container_width=True, hide_index=True)
            else:
                st.info("Sem críticas registradas para essa placa no recorte.")
    else:
        st.info("Nenhuma placa disponível no recorte atual.")

    st.markdown("---")
    st.markdown('<div class="section">Detalhamento da produção da mesa</div>', unsafe_allow_html=True)

    det = viewProd.copy()
    det["DATA_BASE"] = pd.to_datetime(det["DATA_BASE"], errors="coerce").dt.date
    cols = ["DATA_BASE","OS","PLACA","TIPO_USUARIO","USUARIO","STATUS_LAUDO","TEMPO_TOTAL"]
    for c in cols:
        if c not in det.columns:
            det[c] = ""
    det = det[cols].sort_values(["DATA_BASE","OS","TIPO_USUARIO"]).rename(columns={
        "DATA_BASE": "DATA",
        "TIPO_USUARIO": "TIPO USUÁRIO",
        "STATUS_LAUDO": "STATUS LAUDO",
        "TEMPO_TOTAL": "TEMPO TOTAL",
    })
    st.dataframe(det, use_container_width=True, hide_index=True)
    st.caption('<div class="table-note">Produção detalhada da mesa de análise, no período selecionado.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section">Detalhamento das críticas</div>', unsafe_allow_html=True)

    detc_base = viewCrit.copy() if not viewCrit.empty else (dfCrit_mes.copy() if "dfCrit_mes" in locals() else pd.DataFrame())
    if detc_base.empty:
        st.info("Sem críticas para mostrar no recorte atual.")
    else:
        cols_c = ["DATA_CRITICA","OS","PLACA","VISTORIADOR","ANALISTA","STATUS_CRITICA","OBS"]
        for c in cols_c:
            if c not in detc_base.columns:
                detc_base[c] = ""
        detc = detc_base[cols_c].sort_values(["DATA_CRITICA","ANALISTA","VISTORIADOR"]).rename(columns={
            "DATA_CRITICA": "DATA CRÍTICA",
            "STATUS_CRITICA": "STATUS CRÍTICA",
            "OBS": "OBSERVAÇÃO",
        })
        st.dataframe(detc, use_container_width=True, hide_index=True)

        st.caption('<div class="table-note">Cada linha representa uma crítica da mesa (aprovada ou reprovada).</div>', unsafe_allow_html=True)

