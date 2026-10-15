"""
House Price Predictor (Mumbai) - simple local web UI
----------------------------------------------------
Same preprocessing + Random Forest as house_price_prediction.py.
Run:  python house_price_ui.py   -> opens in your browser.
Stop: Ctrl+C in the terminal.
"""

import os
import glob
import json
import zipfile
import time
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---- edit these for your group (shown in the page header) ----
PROJECT = {
    "title": "Mumbai House Price Predictor",
    "subtitle": "Machine Learning mini project",
    "branch": "AIML",
    "college": "DBATU, Lonere",
    "team": ["Member 1", "Member 2", "Member 3"],
}

# everything the program does gets printed AND saved here, the page shows it
LOG = []


def say(text):
    print(text)
    LOG.append(text)


# =====================================================================
# STEP 1: LOAD THE DATASET (looks in this folder and subfolders)
# =====================================================================
def find_csvs(folder):
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    keep = []
    for f in files:
        parts = os.path.relpath(f, folder).lower().split(os.sep)
        # skip the virtual environment, it has sample csv files inside
        if "venv" in parts or ".venv" in parts or "site-packages" in parts:
            continue
        keep.append(f)
    return keep


def load_dataset():
    folder = os.path.dirname(os.path.abspath(__file__))
    csv_files = find_csvs(folder)
    if not csv_files:  # maybe the data is still zipped
        for z in glob.glob(os.path.join(folder, "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder)
        csv_files = find_csvs(folder)
    if not csv_files:
        raise FileNotFoundError("No CSV file found next to this script.")
    say("Loaded dataset: " + os.path.relpath(csv_files[0], folder))
    return pd.read_csv(csv_files[0])


print("=" * 60)
print("HOUSE PRICE PREDICTOR")
print("=" * 60)

df = load_dataset()
raw_rows, raw_cols = df.shape
say(f"Shape: {raw_rows} rows, {raw_cols} columns")

# =====================================================================
# STEP 2: PREPROCESSING (same as the terminal version)
# =====================================================================
# 2.1 target column = the numeric column with "price" in its name
target = None
for col in df.columns:
    if "price" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
        target = col
        break
if target is None:
    raise ValueError("Could not find a price column in the dataset.")
say(f"Target column (what we predict): {target}")

# 2.2 remove duplicates
before = len(df)
df = df.drop_duplicates()
say(f"Removed {before - len(df)} duplicate rows. Rows left: {len(df)}")

# 2.3 missing values: median for numbers, most common value for text
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
say("Missing values: median for numbers, most common value for text")

# 2.4 price is in Crore or Lakh -> convert everything to Lakhs (1 Cr = 100 L)
unit_col = None
for col in df.columns:
    if "unit" in col.lower():
        unit_col = col
        break
if unit_col is not None:
    is_crore = df[unit_col].astype(str).str.strip().str.lower().str.startswith("cr")
    df[target] = np.where(is_crore, df[target] * 100, df[target])
    df = df.drop(columns=[unit_col])
    say("Converted price from Crore/Lakh into Lakhs (1 Cr = 100 L)")

# 2.5 drop ID-like columns
for col in list(df.columns):
    if col != target and ("id" == col.lower() or col.lower().endswith("_id") or col.lower() == "index"):
        df = df.drop(columns=[col])
        say("Dropped ID column: " + col)

# 2.6 drop text columns with too many unique values (like locality)
for col in list(df.columns):
    if col != target and not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 300:
        say(f"Dropped '{col}' (too many unique values: {df[col].nunique()})")
        df = df.drop(columns=[col])

# 2.7 remove outliers using price per area (bottom 1% and top 1%)
area_col = None
for col in df.columns:
    if "area" in col.lower() or "sqft" in col.lower():
        area_col = col
        break
if area_col is not None:
    price_per_area = df[target] / df[area_col]
    low, high = price_per_area.quantile(0.01), price_per_area.quantile(0.99)
    before = len(df)
    df = df[(price_per_area >= low) & (price_per_area <= high)]
    say(f"Removed {before - len(df)} outlier rows (bottom and top 1% of price per {area_col})")

# 2.8 features (X) and target (y)
X = df.drop(columns=[target])
y = df[target]
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
say("Numerical features: " + ", ".join(num_cols))
say("Categorical features: " + ", ".join(cat_cols))
say("Encoding: One-Hot on categorical features. Scaling: not needed for Random Forest")

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)

# =====================================================================
# STEP 3: TRAIN + EVALUATE
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

say(f"Split: {len(X_train)} training rows, {len(X_test)} test rows (80/20)")
print("\nTraining Random Forest, please wait...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
say(f"Model: Random Forest Regressor, 100 trees")
say(f"MAE {mae:.2f} L | MSE {mse:.2f} | RMSE {rmse:.2f} L | R2 {r2:.4f}")

# =====================================================================
# STEP 4: WHAT THE FORM NEEDS TO KNOW
# =====================================================================
fields = []
for col in X.columns:  # exactly the columns the model was trained on
    if col in num_cols:
        fields.append({"name": col, "kind": "number",
                       "min": float(X[col].min()), "max": float(X[col].max())})
    else:
        options = sorted(X[col].unique().tolist())
        fields.append({"name": col, "kind": "choice", "options": [str(o) for o in options]})

# feature importance: add up the one-hot columns that belong to each original feature
names = model.named_steps["preprocess"].get_feature_names_out()
scores = model.named_steps["regressor"].feature_importances_
importance = {c: 0.0 for c in X.columns}
for n, sc in zip(names, scores):
    for c in X.columns:
        if n == f"remainder__{c}" or (c in cat_cols and n.startswith(f"cat__{c}_")):
            importance[c] += float(sc)
            break
importance = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

# 400 random test houses for the actual-vs-predicted chart
pick = np.random.default_rng(0).choice(len(y_test), min(400, len(y_test)), replace=False)
points = [[round(float(y_test.iloc[i]), 1), round(float(y_pred[i]), 1)] for i in pick]

meta = {
    "project": PROJECT,
    "fields": fields,
    "raw_rows": int(raw_rows),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "mae": round(float(mae), 2),
    "mse": round(float(mse), 2),
    "rmse": round(rmse, 2),
    "r2": round(float(r2), 4),
    "importance": importance,
    "points": points,
    "log": LOG,
}

predict_lock = threading.Lock()


def to_frame(rows):
    """Rows of input -> DataFrame with the same columns and dtypes as training."""
    out = pd.DataFrame(rows)[X.columns]
    for col in num_cols:
        if pd.api.types.is_integer_dtype(X[col]):
            out[col] = out[col].round()
        out[col] = out[col].astype(X[col].dtype)
    return out


# a "typical house" (median numbers, most common text) for the what-if bars
baseline = {}
for col in X.columns:
    baseline[col] = float(X[col].median()) if col in num_cols else X[col].mode()[0]


def predict_price(values):
    """Checks the input like the terminal version did, then predicts (in Lakhs)."""
    user_data = {}
    warnings_out = []
    for fd in fields:
        col = fd["name"]
        raw = str(values.get(col, "")).strip()
        if raw == "":
            raise ValueError(f"Fill in {col}.")
        if fd["kind"] == "number":
            try:
                number = float(raw)
            except ValueError:
                raise ValueError(f"{col} must be a number.")
            if number <= 0:
                raise ValueError(f"{col} must be greater than 0.")
            if number < fd["min"] or number > fd["max"]:
                warnings_out.append(f"{col} is outside the range in the data ({fd['min']:g} to {fd['max']:g}), so this estimate is shaky.")
            user_data[col] = number
        else:
            lookup = {o.lower(): o for o in fd["options"]}
            if raw.lower() not in lookup:
                raise ValueError(f"'{raw}' is not a known {col}. Pick one from the list.")
            user_data[col] = lookup[raw.lower()]

    # same columns, same order, same dtypes as the training data
    user_df = to_frame([user_data])

    with predict_lock:
        price = float(model.predict(user_df)[0])  # the real prediction

        # extra 1: every tree gives its own answer, the forest averages them
        Xt = model.named_steps["preprocess"].transform(user_df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        votes = np.array([t.predict(Xt)[0] for t in model.named_steps["regressor"].estimators_])

        # extra 2: what-if, swap ONE input at a time with a typical value
        what_if = []
        for col in X.columns:
            row = dict(user_data)
            row[col] = baseline[col]
            what_if.append(row)
        what_if_preds = model.predict(to_frame(what_if))

    effects = [{"name": c, "value": f"{user_data[c]:g}" if c in num_cols else str(user_data[c]), "delta": float(price - p)}
               for c, p in zip(X.columns, what_if_preds)]
    effects.sort(key=lambda e: abs(e["delta"]), reverse=True)

    t = time.strftime("%H:%M:%S")
    shown = ", ".join(f"{c}={user_data[c]:g}" if c in num_cols else f"{c}={user_data[c]}" for c in X.columns)
    log = [
        f"[{t}] input: {shown}",
        f"[{t}] checked: {len(fields)} fields valid, {len(warnings_out)} warning(s)",
        f"[{t}] one-hot encoded into {Xt.shape[1]} columns",
        f"[{t}] {len(votes)} trees voted: min {votes.min():.1f} L, max {votes.max():.1f} L, std {votes.std():.1f} L",
        f"[{t}] biggest push: {effects[0]['name']} ({effects[0]['delta']:+.1f} L vs a typical house)",
        f"[{t}] estimate: {price:.2f} lakhs",
    ]
    extra = {
        "low": float(np.percentile(votes, 10)),
        "high": float(np.percentile(votes, 90)),
        "std": float(votes.std()),
        "trees": int(len(votes)),
        "cols": int(Xt.shape[1]),
        "effects": effects,
        "log": log,
    }
    return price, warnings_out, extra


# =====================================================================
# STEP 5: THE PAGE
# =====================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mumbai House Price Predictor</title>
<style>
  :root {
    --bg: #0e1217; --panel: #161b22; --edge: #2a323d; --ink: #e7e9ec;
    --soft: #8d96a3; --taxi: #f4c20d; --blue: #4fb3e8; --bad: #ff8a80;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 22px 60px; }

  header { border-left: 6px solid var(--taxi); padding: 4px 0 4px 18px; margin-bottom: 26px; }
  h1 { margin: 0; font: 700 40px/1.05 Bahnschrift, "Arial Narrow", sans-serif; letter-spacing: .01em; }
  .tag { color: var(--soft); margin: 6px 0 12px; }
  .team { display: flex; flex-wrap: wrap; gap: 8px; }
  .team span { border: 1px solid var(--edge); background: var(--panel); padding: 2px 12px; font-size: 14px; }

  .grid { display: grid; grid-template-columns: 400px 1fr; gap: 18px; align-items: start; }
  @media (max-width: 880px) { .grid { grid-template-columns: 1fr; } }
  .col { display: grid; gap: 18px; }

  section { background: var(--panel); border: 1px solid var(--edge); padding: 18px 20px 20px; }
  h2 { margin: 0 0 14px; font: 700 20px Bahnschrift, "Arial Narrow", sans-serif; letter-spacing: .02em; }
  h2 small { font: 400 13px "Segoe UI", sans-serif; color: var(--soft); margin-left: 8px; }

  .f { margin-bottom: 16px; }
  label { display: block; font-size: 14px; color: var(--soft); margin-bottom: 5px; }
  .hint { color: var(--soft); font-size: 12.5px; margin-top: 3px; }
  input[type=text], input[type=number] {
    width: 100%; font: inherit; color: var(--ink); background: var(--bg);
    border: 1px solid var(--edge); border-radius: 3px; padding: 8px 10px; outline: none;
  }
  input:focus { border-color: var(--taxi); }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; }
  .pills button {
    font: inherit; font-size: 14px; color: var(--ink); background: var(--bg); cursor: pointer;
    border: 1px solid var(--edge); border-radius: 3px; padding: 6px 12px;
  }
  .pills button[aria-pressed=true] { background: var(--taxi); color: #14161a; border-color: var(--taxi); font-weight: 600; }
  button:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
  .go {
    width: 100%; font: 700 18px Bahnschrift, "Arial Narrow", sans-serif; letter-spacing: .03em;
    background: var(--taxi); color: #14161a; border: 0; border-radius: 3px; padding: 12px; cursor: pointer;
  }
  .err { color: var(--bad); margin-top: 10px; min-height: 1.4em; font-size: 14px; }
  #out { display: none; margin-top: 12px; border: 1px dashed var(--taxi); padding: 14px 16px; }
  #out.show { display: block; }
  #big { font: 700 46px/1 Bahnschrift, "Arial Narrow", sans-serif; color: var(--taxi); }
  #lakh { color: var(--soft); margin-top: 4px; }
  #warn { color: var(--bad); font-size: 13.5px; margin-top: 8px; }

  .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  @media (max-width: 520px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
  .tile { border: 1px solid var(--edge); background: var(--bg); padding: 10px 12px; }
  .tile b { display: block; font: 700 26px/1.1 Bahnschrift, "Arial Narrow", sans-serif; color: var(--taxi); }
  .tile span { font-size: 13px; color: var(--soft); }

  canvas { width: 100%; height: 300px; display: block; background: var(--bg); border: 1px solid var(--edge); }
  .bar { display: grid; grid-template-columns: 90px 1fr 52px; gap: 10px; align-items: center; margin-bottom: 8px; font-size: 14px; }
  .bar i { display: block; height: 14px; background: var(--blue); }
  .bar em { font-style: normal; color: var(--soft); text-align: right; }

  pre {
    margin: 0; background: #07090c; border: 1px solid var(--edge); padding: 14px 16px; overflow-x: auto;
    font: 13px/1.6 Consolas, "Courier New", monospace; color: #9fe3a8;
  }
  .model { color: var(--soft); font-size: 14px; margin: 0 0 12px; }
  .eff { display: grid; grid-template-columns: 150px 1fr 76px; gap: 10px; align-items: center; margin-bottom: 8px; font-size: 14px; }
  .eff .n { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .eff em { font-style: normal; text-align: right; color: var(--soft); }
  .trk { position: relative; height: 14px; }
  .trk::before { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--soft); }
  .trk i { position: absolute; top: 0; height: 100%; display: block; }
  .now { margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--taxi); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title">House Price Predictor</h1>
    <div class="tag" id="tag"></div>
    <div class="team" id="team"></div>
  </header>

  <div class="grid">
    <div class="col">
      <section>
        <h2>Predict a price</h2>
        <form id="form" autocomplete="off">
          <div id="fields"></div>
          <button class="go" type="submit">Predict price</button>
          <div class="err" id="err" role="alert"></div>
        </form>
        <div id="out" aria-live="polite">
          <div id="big"></div>
          <div id="lakh"></div>
          <div id="warn"></div>
        </div>
      </section>
    </div>

    <div class="col">
      <section>
        <h2>Model results <small>on the 20% test split</small></h2>
        <p class="model" id="model"></p>
        <div class="tiles" id="tiles"></div>
        <div class="now" id="nowBox" hidden>
          <p class="model">Your house, as seen by the 100 trees</p>
          <div class="tiles" id="nowTiles"></div>
        </div>
      </section>

      <section>
        <h2>Actual vs predicted <small>price in lakhs, 400 test houses</small></h2>
        <canvas id="chart"></canvas>
      </section>

      <section>
        <h2><span id="impT">What the model looks at</span><small id="impS">feature importance from training</small></h2>
        <div id="imp"></div>
      </section>
    </div>
  </div>

  <section style="margin-top:18px">
    <h2>Pipeline log <small>what the program did to the data</small></h2>
    <pre id="log"></pre>
  </section>
</div>

<script>
const $ = s => document.querySelector(s);
const state = {};
let meta = null;

fetch('/meta').then(r => r.json()).then(m => { meta = m; build(); stats(); draw(); });
window.onresize = () => meta && draw();

function nice(n) {
  n = n.replace(/_/g, ' ');
  return n.length <= 3 ? n.toUpperCase() : n[0].toUpperCase() + n.slice(1);
}
const fmt = n => n.toLocaleString(undefined, { maximumFractionDigits: 2 });

function build() {
  const p = meta.project;
  $('#title').textContent = p.title;
  $('#tag').textContent = p.subtitle + ' - ' + p.branch + ', ' + p.college;
  p.team.forEach(t => { const s = document.createElement('span'); s.textContent = t; $('#team').appendChild(s); });

  const box = $('#fields');
  meta.fields.forEach(fd => {
    const wrap = document.createElement('div'); wrap.className = 'f';
    const lab = document.createElement('label'); lab.textContent = nice(fd.name); wrap.appendChild(lab);
    if (fd.kind === 'number') {
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = 'any'; inp.min = '0';
      inp.id = lab.htmlFor = 'f_' + fd.name;
      inp.oninput = () => state[fd.name] = inp.value;
      wrap.appendChild(inp);
      const h = document.createElement('div'); h.className = 'hint';
      h.textContent = 'In the data: ' + fd.min + ' to ' + fd.max; wrap.appendChild(h);
    } else if (fd.options.length <= 10) {
      const row = document.createElement('div'); row.className = 'pills';
      fd.options.forEach(o => {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = o; b.setAttribute('aria-pressed', 'false');
        b.onclick = () => {
          row.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', 'false'));
          b.setAttribute('aria-pressed', 'true'); state[fd.name] = o;
        };
        row.appendChild(b);
      });
      wrap.appendChild(row);
    } else {
      const inp = document.createElement('input');
      inp.type = 'text'; inp.setAttribute('list', 'l_' + fd.name);
      inp.id = lab.htmlFor = 'f_' + fd.name; inp.placeholder = 'Start typing a name';
      inp.oninput = () => state[fd.name] = inp.value;
      const dl = document.createElement('datalist'); dl.id = 'l_' + fd.name;
      fd.options.forEach(o => { const op = document.createElement('option'); op.value = o; dl.appendChild(op); });
      wrap.appendChild(inp); wrap.appendChild(dl);
      const h = document.createElement('div'); h.className = 'hint';
      h.textContent = fd.options.length + ' options in the data'; wrap.appendChild(h);
    }
    box.appendChild(wrap);
  });
}

function stats() {
  $('#model').textContent = 'Random Forest Regressor, 100 trees. Trained on ' + fmt(meta.train_rows) +
    ' rows, tested on ' + fmt(meta.test_rows) + '.';
  const t = [
    [meta.r2, 'R2 score'], [fmt(meta.mae) + ' L', 'MAE'], [fmt(meta.rmse) + ' L', 'RMSE'],
    [fmt(meta.mse), 'MSE'], [fmt(meta.raw_rows), 'rows in the CSV'], [meta.fields.length, 'input features']
  ];
  t.forEach(([v, l]) => {
    const d = document.createElement('div'); d.className = 'tile';
    d.innerHTML = '<b></b><span></span>'; d.firstChild.textContent = v; d.lastChild.textContent = l;
    $('#tiles').appendChild(d);
  });
  const mx = Math.max(...meta.importance.map(i => i[1]));
  meta.importance.forEach(([n, v]) => {
    const r = document.createElement('div'); r.className = 'bar';
    r.innerHTML = '<span></span><div><i></i></div><em></em>';
    r.children[0].textContent = nice(n);
    r.children[1].firstChild.style.width = (v / mx * 100) + '%';
    r.children[2].textContent = (v * 100).toFixed(1) + '%';
    $('#imp').appendChild(r);
  });
  $('#log').textContent = meta.log.map(l => '> ' + l).join('\n');
}

let lastPred = null;

function addLog(lines) {
  const el = $('#log');
  el.textContent += '\n' + lines.map(l => '> ' + l).join('\n');
  el.scrollTop = el.scrollHeight;
}

function showNow(d) {
  const box = $('#nowTiles'); box.innerHTML = '';
  [[fmt(d.lakhs) + ' L', 'estimate'], [fmt(d.low) + ' L', 'low (10% of trees below)'],
   [fmt(d.high) + ' L', 'high (90% of trees below)'], [fmt(d.std) + ' L', 'tree disagreement (std)'],
   [d.trees, 'trees that voted'], [d.cols, 'columns after encoding']].forEach(([v, l]) => {
    const t = document.createElement('div'); t.className = 'tile';
    t.innerHTML = '<b></b><span></span>'; t.firstChild.textContent = v; t.lastChild.textContent = l;
    box.appendChild(t);
  });
  $('#nowBox').hidden = false;
}

function showEffects(d) {
  $('#impT').textContent = 'What moved your estimate';
  $('#impS').textContent = 'change vs a typical house, one input at a time';
  const box = $('#imp'); box.innerHTML = '';
  const mx = Math.max(...d.effects.map(e => Math.abs(e.delta)), 0.01);
  d.effects.forEach(e => {
    const r = document.createElement('div'); r.className = 'eff';
    r.innerHTML = '<span class="n"></span><div class="trk"><i></i></div><em></em>';
    r.children[0].textContent = nice(e.name) + ' ' + e.value;
    const bar = r.children[1].firstChild, w = Math.abs(e.delta) / mx * 50;
    bar.style.width = w + '%';
    bar.style.left = e.delta >= 0 ? '50%' : (50 - w) + '%';
    bar.style.background = e.delta >= 0 ? 'var(--taxi)' : 'var(--bad)';
    r.children[2].textContent = (e.delta >= 0 ? '+' : '') + e.delta.toFixed(1) + ' L';
    box.appendChild(r);
  });
}

function draw() {
  const c = $('#chart'), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext('2d'); g.scale(dpr, dpr);
  const pad = { l: 52, r: 14, t: 12, b: 36 };
  const lim = Math.max(...meta.points.flat(), lastPred ? lastPred.high : 0) * 1.03;
  const X = v => pad.l + v / lim * (W - pad.l - pad.r);
  const Y = v => H - pad.b - v / lim * (H - pad.t - pad.b);
  g.font = '12px Segoe UI'; g.fillStyle = '#8d96a3'; g.strokeStyle = '#2a323d'; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lim * i / 4;
    g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(W - pad.r, Y(v)); g.stroke();
    g.textAlign = 'right'; g.fillText(Math.round(v), pad.l - 6, Y(v) + 4);
    g.textAlign = 'center'; g.fillText(Math.round(v), X(v), H - pad.b + 16);
  }
  g.fillText('actual', W / 2, H - 6);
  g.save(); g.translate(12, H / 2); g.rotate(-Math.PI / 2); g.fillText('predicted', 0, 0); g.restore();
  g.setLineDash([6, 5]); g.strokeStyle = '#f4c20d'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(lim), Y(lim)); g.stroke(); g.setLineDash([]);
  g.fillStyle = 'rgba(79,179,232,.55)';
  meta.points.forEach(([a, p]) => { g.beginPath(); g.arc(X(a), Y(p), 2.6, 0, 6.283); g.fill(); });
  if (lastPred) {  // your prediction: shaded range + line
    g.fillStyle = 'rgba(244,194,13,.16)';
    g.fillRect(pad.l, Y(lastPred.high), W - pad.l - pad.r, Y(lastPred.low) - Y(lastPred.high));
    g.strokeStyle = '#f4c20d'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(pad.l, Y(lastPred.lakhs)); g.lineTo(W - pad.r, Y(lastPred.lakhs)); g.stroke();
    g.fillStyle = '#f4c20d'; g.textAlign = 'right';
    g.fillText('your estimate ' + Math.round(lastPred.lakhs) + ' L', W - pad.r - 4, Y(lastPred.lakhs) - 6);
  }
}

$('#form').onsubmit = async e => {
  e.preventDefault();
  $('#err').textContent = '';
  const res = await fetch('/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state)
  });
  const data = await res.json();
  if (data.error) {
    $('#out').classList.remove('show'); $('#err').textContent = data.error;
    addLog(['[' + new Date().toTimeString().slice(0, 8) + '] rejected: ' + data.error]);
    return;
  }
  const L = data.lakhs;
  $('#big').textContent = L >= 100 ? 'Rs. ' + (L / 100).toFixed(2) + ' Cr' : 'Rs. ' + L.toFixed(2) + ' L';
  $('#lakh').textContent = L.toFixed(2) + ' lakhs';
  $('#warn').textContent = data.warnings.join(' ');
  $('#out').classList.add('show');
  lastPred = data;
  showNow(data); showEffects(data); addLog(data.log); draw();
};
</script>
</body>
</html>
"""


# =====================================================================
# STEP 6: TINY SERVER
# =====================================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def reply(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, PAGE, "text/html")
        elif self.path == "/meta":
            self.reply(200, json.dumps(meta), "application/json")
        else:
            self.reply(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self.reply(404, "not found", "text/plain")
        try:
            size = int(self.headers.get("Content-Length", 0))
            values = json.loads(self.rfile.read(size) or b"{}")
            price, warns, extra = predict_price(values)
            out = {"lakhs": price, "warnings": warns, **extra}
        except ValueError as err:
            out = {"error": str(err)}
        except Exception:
            out = {"error": "Something went wrong. Check the values and try again."}
        self.reply(200, json.dumps(out), "application/json")


server = None
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        Handler
    )

    url = f"http://localhost:{port}"

    print(f"Server running on port {port}")

    if not os.environ.get("PORT"):
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()