"""Zip ONLY the subset images so they can be uploaded to Google Drive for Colab (Notebook 02).
Run on your PC (inside the Notebooks folder or anywhere):
    python make_images_zip.py "C:\\path\\to\\DataSet"
Creates  DataSet/outputs/subset_images.zip  (~300-400 MB). Upload it together with subset.parquet to Drive/MultiCaRe/."""
import sys, os, zipfile
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "../DataSet"
out_dir = os.path.join(root, "outputs")
df = pd.read_parquet(os.path.join(out_dir, "subset.parquet"), columns=["file"])
zpath = os.path.join(out_dir, "subset_images.zip")
missing = 0
with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:   # webp is already compressed
    for i, f in enumerate(df["file"]):
        rel = f"{f[:4]}/{f[:5]}/{f}"                                       # rule from the data dictionary
        p = os.path.join(root, f[:4], f[:5], f)
        if os.path.exists(p): z.write(p, rel)
        else: missing += 1
        if i % 2000 == 0: print(i, "/", len(df))
print("done:", zpath, "| size MB:", round(os.path.getsize(zpath) / 1e6, 1), "| missing files:", missing)
print("Upload to Google Drive folder MultiCaRe/ :", zpath, "and", os.path.join(out_dir, "subset.parquet"))
