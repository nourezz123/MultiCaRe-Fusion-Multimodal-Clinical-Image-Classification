"""Run:  python profile_multicare.py "C:\\path\\to\\DataSet"
Creates DataSet/profile_out/report.txt and sample.csv (both small, send them to Claude)."""
import sys, os, io, random
import pandas as pd, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
out = os.path.join(root, "profile_out"); os.makedirs(out, exist_ok=True)
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")

pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 80)
cl = pd.read_csv(os.path.join(root, "captions_and_labels.csv"))
ci = pd.read_parquet(os.path.join(root, "case_images.parquet"))
cs = pd.read_parquet(os.path.join(root, "cases.parquet"))

log("== captions_and_labels", cl.shape); log(cl.dtypes.to_string())
log("\nMissing rate:\n", cl.isna().mean().sort_values(ascending=False).round(3).to_string())
log("\nFirst 3 rows:\n", cl.head(3).T.to_string())
for c in ["image_type", "image_subtype", "radiology_region", "radiology_region_granular", "radiology_view"]:
    vc = cl[c].value_counts(dropna=False)
    log(f"\n-- {c}: {cl[c].nunique()} classes; imbalance max/min = {vc.max()/max(vc.min(),1):.0f}")
    log(vc.to_string())
for c in [c for c in cl.columns if "supervised" in c]:
    s = cl[c].dropna().astype(str)
    log(f"\n-- {c}: non-null={len(s)}; sample values:", s.head(3).tolist())
log("\nCaption length (chars):\n", cl["caption"].fillna("").str.len().describe().round(1).to_string())
log("Empty/NaN captions:", int(cl["caption"].isna().sum() + (cl["caption"].fillna("").str.strip() == "").sum()))
log("Duplicate captions:", int(cl["caption"].duplicated().sum()))
log("Patients:", cl["patient_id"].nunique(), "| main_images:", cl["main_image"].nunique())
log("Images per patient:\n", cl.groupby("patient_id").size().describe().round(1).to_string())
log("Split sub-images (image_component != undivided):", int((cl["image_component"] != "undivided").sum()))
log("File size (bytes):\n", cl["file_size"].describe().round(0).to_string())

import json
def flatten(df, col):
    x = df[["article_id", col]].explode(col).dropna(subset=[col]).reset_index(drop=True)
    first = x[col].iloc[0] if len(x) else None
    if isinstance(first, str):
        try: x[col] = x[col].map(json.loads); first = x[col].iloc[0]
        except Exception: pass
    if isinstance(first, dict):
        return pd.concat([x[["article_id"]], pd.json_normalize(x[col].tolist())], axis=1)
    return x

def peek(df, name):
    log("\n==", name, df.shape, df.columns.tolist()); log(df.dtypes.to_string())
    for c in df.columns:
        v = df[c].iloc[0]
        log(f"  first value of '{c}': type={type(v).__name__} ->", repr(v)[:700])

peek(ci, "case_images (raw)"); peek(cs, "cases (raw)")
try:
    if "case_text" not in cs.columns:
        listcol = [c for c in cs.columns if c != "article_id"][0]
        cs = flatten(cs, listcol); peek(cs, "cases (flattened)")
    if "caption" not in ci.columns:
        listcol = [c for c in ci.columns if c != "article_id"][0]
        ci = flatten(ci, listcol); peek(ci, "case_images (flattened)")
    if "age" in cs.columns:
        log("age missing:", round(cs["age"].isna().mean(), 3)); log(pd.to_numeric(cs["age"], errors="coerce").describe().round(1).to_string())
    if "gender" in cs.columns: log(cs["gender"].value_counts(dropna=False).to_string())
    if "case_text" in cs.columns: log("case_text length:\n", cs["case_text"].fillna("").str.len().describe().round(0).to_string())
    if "case_id" in cs.columns: log("patient_id in cases.case_id:", round(cl["patient_id"].isin(cs["case_id"]).mean(), 4))
except Exception as e:
    log("!! cases/case_images flatten failed:", repr(e))

# image path check (rule from data dictionary)
def path(f): return os.path.join(root, f[:4], f[:5], f)
samp = cl.sample(min(500, len(cl)), random_state=0)
ex = samp["file"].map(lambda f: os.path.exists(path(f)))
log("\nImage files found (500 random):", int(ex.sum()), "/", len(samp))
if ex.sum() == 0:
    log("Example expected path:", path(samp["file"].iloc[0]))
    log("Top folders in root:", [d for d in os.listdir(root)][:20])

# stratified sample for Claude (<= 100 per image_subtype), short text
parts = [g.sample(min(len(g), 100), random_state=0) for _, g in cl.groupby("image_subtype", dropna=False)]
sm = pd.concat(parts)
try:
    cs1 = cs.drop_duplicates("case_id")[["case_id", "age", "gender", "case_text"]].copy()
    cs1["case_text"] = cs1["case_text"].fillna("").str.slice(0, 400)
    sm = sm.merge(cs1, left_on="patient_id", right_on="case_id", how="left")
except Exception as e:
    log("!! could not merge cases into sample:", repr(e))
sm["caption"] = sm["caption"].fillna("").str.slice(0, 400)
sm["file_exists"] = sm["file"].map(lambda f: os.path.exists(path(f)))
sm.to_csv(os.path.join(out, "sample.csv"), index=False)
log("\nsample.csv rows:", len(sm), "size MB:", round(os.path.getsize(os.path.join(out, "sample.csv"))/1e6, 2))
open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write(buf.getvalue())
print("\nDone. Send report.txt and sample.csv from", out)