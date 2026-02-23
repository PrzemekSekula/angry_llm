import pandas as pd
import re
import matplotlib.pyplot as plt

# === 1. Wczytaj CSV ===
df = pd.read_csv("../log/ab_prompts_20260219_125648.csv")

# === 2. Wyciągnij liczby z OFFER: A=x, B=y ===
def extract_offer(text):
    if pd.isna(text):
        return None, None
    match = re.search(r"A\s*=\s*(\d+).*?B\s*=\s*(\d+)", str(text))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

df[["A_share", "B_share"]] = df["response_out"].apply(
    lambda x: pd.Series(extract_offer(x))
)

# Usuń wiersze bez oferty (np. REJECT bez liczby)
df_offers = df.dropna(subset=["A_share", "B_share"])

# === 3. Wykres ===
plt.figure()

plt.plot(df_offers["iteration"], df_offers["A_share"], marker="o", label="A share")
plt.plot(df_offers["iteration"], df_offers["B_share"], marker="o", label="B share")

plt.xlabel("Iteration")
plt.ylabel("Credits")
plt.title("Negotiation Dynamics")
plt.legend()
plt.grid(True)

plt.show()


