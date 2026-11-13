#!/usr/bin/env python3
"""Download identified PDFs. Python 3 standard library only; run with Internet access."""
import csv, hashlib, json, time, urllib.request
from pathlib import Path
BASE = Path(__file__).resolve().parent
refs = json.loads((BASE / "references.json").read_text(encoding="utf-8"))
dest = BASE / "articles"
dest.mkdir(exist_ok=True)
results = []
for r in refs:
    row = {"key": r["key"], "title": r["title"], "url": r["pdf_url"], "status": "no_pdf_url", "sha256": "", "error": ""}
    if r["pdf_url"]:
        target = dest / (r["key"] + ".pdf")
        try:
            req = urllib.request.Request(r["pdf_url"], headers={"User-Agent": "BibliographyDownloader/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            if not data.lstrip().startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF (possible login, captcha, or HTML error).")
            target.write_bytes(data)
            row.update(status="downloaded", sha256=hashlib.sha256(data).hexdigest())
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        time.sleep(2)
    results.append(row)
    print(row["key"], row["status"])
with (BASE / "download_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
print("Downloaded:", sum(r["status"] == "downloaded" for r in results), "/", len(refs))
