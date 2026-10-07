"""Download the 68 recommended Tufts fNIRS2MW 30-second window CSVs from public Box.

The public folder is documented at:
https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html

Files are resumed/skipped if already present and look like CSV data. HTML/error
pages are rejected rather than silently entering the training set.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

SHARE = "7l8rz3tpos1il637kmhn57mzacdldyv4"
FILE_IDS = {
    1:"854967732957",5:"854965446153",7:"854968407733",13:"854950541251",14:"854964318511",15:"854965297638",
    20:"854968177342",21:"854954594726",22:"854968438108",23:"854964770629",24:"854968213825",25:"854968270706",
    27:"854965152288",28:"854964296021",29:"854967299524",31:"854968292474",32:"854967653331",34:"854964698843",
    35:"854964908271",36:"854964249558",37:"854968217310",38:"854965223519",40:"854965762785",42:"854964401792",
    43:"854965335014",44:"854967722557",45:"854967475749",46:"854964585104",47:"854964174488",48:"854964473111",
    49:"854967382149",51:"854965725154",52:"854965282187",54:"854967848579",55:"854964656977",56:"854968310283",
    57:"854967406685",58:"854967532074",60:"854967845425",61:"854965008328",62:"854963182213",63:"854964104921",
    64:"854965063814",65:"854966298096",68:"854968046542",69:"854967787856",70:"854964984022",71:"854966239921",
    72:"854967829395",73:"854964748595",74:"854965295097",75:"854967697156",76:"854967385988",78:"854962496752",
    79:"854964863686",80:"854965353014",81:"854965052009",82:"854964683151",83:"854967539698",84:"854968612501",
    85:"854968727949",86:"854967869218",91:"854966384372",92:"854968781267",93:"854965626153",94:"854968906504",
    95:"854964962293",97:"854968069227",
}

def looks_like_csv(path: Path) -> bool:
    try:
        if path.stat().st_size < 1000:
            return False
        with path.open("rb") as f:
            head = f.read(4096).lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title>box" in head:
            return False
        text = head.decode("utf-8", "ignore")
        return "chunk" in text and "label" in text and "," in text
    except Exception:
        return False


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0 WITHIN-research-prototype/1.0"})
    with urlopen(req, timeout=120) as r, tmp.open("wb") as w:
        ctype = (r.headers.get("content-type") or "").lower()
        # Box can sometimes return an HTML landing/error page with HTTP 200.
        first = r.read(4096)
        if "text/html" in ctype or first.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise RuntimeError("Box returned HTML instead of CSV")
        w.write(first)
        while True:
            b = r.read(1024 * 1024)
            if not b: break
            w.write(b)
    if not looks_like_csv(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded file failed CSV sanity check")
    tmp.replace(dest)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="tufts_30s")
    ap.add_argument("--min-subjects", type=int, default=20)
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ok=[]; failed=[]
    for i,(sid,fid) in enumerate(sorted(FILE_IDS.items()),1):
        dest=out/f"sub_{sid}.csv"
        if looks_like_csv(dest):
            print(f"[{i:02d}/68] sub_{sid}: already present")
            ok.append(sid); continue
        dest.unlink(missing_ok=True)
        urls=[
            f"https://tufts.app.box.com/s/{SHARE}/content/{fid}?dl=1",
            f"https://tufts.app.box.com/index.php?rm=box_download_shared_file&shared_name={SHARE}&file_id=f_{fid}",
        ]
        err=None
        for u in urls:
            try:
                print(f"[{i:02d}/68] sub_{sid}: downloading…", flush=True)
                fetch(u,dest); ok.append(sid); err=None; break
            except Exception as e:
                err=e; dest.unlink(missing_ok=True)
                time.sleep(.4)
        if err is not None:
            failed.append((sid,str(err)))
            print(f"  FAILED sub_{sid}: {err}", file=sys.stderr)
    print(f"\nTufts download: {len(ok)}/68 participants ready in {out}")
    if failed:
        print("Missing/failed:", ", ".join(f"sub_{s}" for s,_ in failed))
    # Machine-readable count for shell/notebook callers.
    (out/"DOWNLOAD_STATUS.txt").write_text(f"ready={len(ok)}\nfailed={len(failed)}\n")
    return 0 if len(ok) >= a.min_subjects else 2

if __name__ == "__main__":
    raise SystemExit(main())
