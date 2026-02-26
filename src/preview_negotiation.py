import re
import streamlit as st
import pandas as pd
from pathlib import Path



st.set_page_config(layout="wide")
st.title("LLM Negotiation Viewer (A/B clean)")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
LOG_DIR = Path("../log")  # jeśli odpalasz streamlit z src/. Jak odpalasz z root, daj "log"
def get_csv_files():
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob("*.csv"), reverse=True)

csv_files = get_csv_files()

st.subheader("Select log file")

if not csv_files:
    st.warning(f"No CSV files found in: {LOG_DIR.resolve()}")
    st.stop()

csv_names = [p.name for p in csv_files]

# zapamiętaj wybrany plik w session_state
if "selected_log" not in st.session_state:
    st.session_state.selected_log = csv_names[0]

selected_name = st.selectbox(
    "Choose CSV file",
    options=csv_names,
    index=csv_names.index(st.session_state.selected_log)
    if st.session_state.selected_log in csv_names
    else 0,
)

# aktualizacja state
st.session_state.selected_log = selected_name

file_path = LOG_DIR / selected_name
st.caption(f"Loaded file: {file_path.resolve()}")

df = load_data(str(file_path))


# csv_files = sorted(LOG_DIR.glob("*.csv"), reverse=True)

# if not csv_files:
#     st.error(f"Nie znaleziono plików CSV w katalogu: {LOG_DIR.resolve()}")
#     st.stop()

# # Lista nazw do selectbox
# csv_names = [p.name for p in csv_files]

# selected_name = st.selectbox(
#     "Wybierz plik log (*.csv) do podglądu",
#     options=csv_names,
#     index=0,  # domyślnie najnowszy (bo reverse=True)
# )

# file_path = str(LOG_DIR / selected_name)
# st.caption(f"Ładuję: {file_path}")


# # ---- Wczytanie CSV ----
# @st.cache_data
# def load_data(path: str) -> pd.DataFrame:
#     return pd.read_csv(path)

# file_path = "../log/ab_prompts_20260219_125648.csv"
# df = load_data(file_path)

# ---- Helpers: wyciągnij tylko "negocjacyjny" prompt (ostatni HUMAN) ----
def extract_last_human_block(prompt_in: str) -> str:
    """
    Z prompt_in w formacie:
      [SYSTEM] ... [AI] ... [HUMAN] <TU>
    wyciąga ostatni blok [HUMAN] jako 'czysty' input negocjacyjny.
    """
    if pd.isna(prompt_in):
        return ""
    text = str(prompt_in)

    # znajdź wszystkie bloki [HUMAN] ... (do następnego [SYSTEM]/[AI]/[HUMAN] lub końca)
    blocks = re.findall(
        r"\[HUMAN\]\s*\n(.*?)(?=\n\[(?:SYSTEM|AI|HUMAN)\]\s*\n|\Z)",
        text,
        flags=re.DOTALL
    )
    if not blocks:
        return ""

    last = blocks[-1].strip()
    return last

def clean_for_table(s: str, n: int = 250) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

# ---- Zrób kolumnę clean_in ----
df = df.copy()
df["clean_in"] = df["prompt_in"].apply(extract_last_human_block)

# ---- Zbuduj tabelę per iteracja: A_in/A_out/B_in/B_out ----
A = df[df["speaker"] == "A"][["iteration", "clean_in", "response_out"]].rename(
    columns={"clean_in": "A_in", "response_out": "A_out"}
)
B = df[df["speaker"] == "B"][["iteration", "clean_in", "response_out"]].rename(
    columns={"clean_in": "B_in", "response_out": "B_out"}
)

merged = pd.merge(A, B, on="iteration", how="outer").sort_values("iteration")
# ---- Dołącz EMO (anger_state / anger_intensity) per iteracja ----
EMO = df[df["speaker"] == "EMO"][["iteration", "anger_intensity", "anger_state"]]

# jeśli zdarzy się więcej niż 1 EMO na iterację (np. bug), bierzemy ostatni
EMO = EMO.sort_values("iteration").groupby("iteration", as_index=False).tail(1)

merged = pd.merge(merged, EMO, on="iteration", how="left").sort_values("iteration")

# ---- Widok: preview vs full ----
show_preview = st.toggle("Preview mode (short text)", value=True)

view = merged.copy()
if show_preview:
    for col in ["A_in", "A_out", "B_in", "B_out", "anger_state"]:
        view[col] = view[col].apply(lambda x: clean_for_table(x, n=220))

st.dataframe(view, use_container_width=True)

# ---- Inspekcja: pełne pola dla wybranej iteracji ----
st.divider()
st.subheader("Inspect iteration")

iters = [int(x) for x in merged["iteration"].dropna().unique()]
iters = sorted(iters)

selected_it = st.selectbox("Iteration", iters, index=0)
row = merged[merged["iteration"] == selected_it].iloc[0]
st.info(f"Anger: {row.get('anger_intensity', '')} | {row.get('anger_state', '')}")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### A input (clean)")
    st.code(row.get("A_in", "") or "", wrap_lines=True, language=None)
    st.markdown("### A output")
    st.code(row.get("A_out", "") or "", wrap_lines=True, language=None)

with c2:
    st.markdown("### B input (clean)")
    st.code(row.get("B_in", "") or "", wrap_lines=True, language=None)
    st.markdown("### B output")
    st.code(row.get("B_out", "") or "", wrap_lines=True, language=None)



# =========================
# FULL HISTORY TABLE (per iteration)
# =========================

st.divider()
st.subheader("Full history per iteration (raw prompt_in)")

A_full = df[df["speaker"] == "A"][["iteration", "prompt_in"]].rename(
    columns={"prompt_in": "history_a_full"}
)
B_full = df[df["speaker"] == "B"][["iteration", "prompt_in"]].rename(
    columns={"prompt_in": "history_b_full"}
)

merged_full = pd.merge(A_full, B_full, on="iteration", how="outer").sort_values("iteration")

show_full_preview = st.toggle("Preview full history (short text)", value=True)

full_view = merged_full.copy()
if show_full_preview:
    for col in ["history_a_full", "history_b_full"]:
        full_view[col] = full_view[col].apply(lambda x: clean_for_table(x, n=350))

st.dataframe(full_view, use_container_width=True)

# =========================
# Inspect: Full history for selected iteration
# =========================

st.divider()
st.subheader("Inspect full history for selected iteration")

row_full = merged_full[merged_full["iteration"] == selected_it].iloc[0]

c3, c4 = st.columns(2)

with c3:
    st.markdown("### history_a (full prompt_in)")
    st.code(row_full.get("history_a_full", "") or "", wrap_lines=True, language=None)

with c4:
    st.markdown("### history_b (full prompt_in)")
    st.code(row_full.get("history_b_full", "") or "", wrap_lines=True, language=None)
