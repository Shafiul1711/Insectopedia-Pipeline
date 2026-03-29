#!/usr/bin/env python3
"""
iNaturalist downloader for selected species with:
- research-grade observations only (default) or verifiable+research grade per species
- per-species count targets
- optional per-species skip
- CSV logging for references/documentation
- page usage logging
- graceful handling when fewer images exist than requested

Install:
    pip install requests

Run:
    python3 inat_downloader_final.py
"""

from __future__ import annotations

import csv
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import requests

API_URL = "https://api.inaturalist.org/v1/observations"

OUTPUT_ROOT = Path("ToAdd")
REF_CSV = Path("inat_download_references.csv")
PAGE_LOG = Path("page_log.txt")

REQUEST_TIMEOUT = 30
PER_PAGE = 200
API_SLEEP_SECONDS = 1.1
IMAGE_SLEEP_SECONDS = 0.15
IMAGE_SIZE = "large"  # square / small / medium / large / original

# Optional: restrict photo licenses if desired, e.g. "cc0,cc-by,cc-by-nc"
PHOTO_LICENSES: Optional[str] = None

# quality_grade options per species:
#   "research"            → research grade only (default)
#   "research,verifiable" → both research and verifiable grades
SPECIES_CONFIG = [
    #{"name": "striped flea beetle",          "count": 50,  "skip": 450, "replace": False},
    #{"name": "black blister beetle",         "count": 50,  "skip": 450, "replace": False},
    #{"name": "Plum Curculio",               "count": 400, "skip": 0,   "replace": False},
    #{"name": "tarnished plant bug",          "count": 50,  "skip": 450, "replace": True},
    #{"name": "codling moth",                "count": 400, "skip": 0,   "replace": False},
    #{"name": "grape berry moth",            "count": 400, "skip": 0,   "replace": False},
    #{"name": "diamondback moth",            "count": 400, "skip": 0,   "replace": False},
    #{"name": "grape flea beetle",            "count": 50,  "skip": 450, "replace": True},
    #{"name": "black blister beetle",         "count": 50,  "skip": 450, "replace": True},
    #{"name": "brown marmorated stink bug",   "count": 50,  "skip": 450, "replace": True},
    #{"name": "colorado potato beetle",       "count": 50,  "skip": 450, "replace": True},
    #{"name": "green stink bug",              "count": 50,  "skip": 450, "replace": True},
    #{"name": "striped blister beetle",       "count": 50,  "skip": 450, "replace": True},
    #{"name": "striped flea beetle",          "count": 50,  "skip": 450, "replace": True},
    #{"name": "plum Curculio",               "count": 400, "skip": 0,   "replace": False},
    #{"name": "alfalfa weevil",               "count": 50,  "skip": 600, "replace": False},
    #{"name": "four lined plant bug",         "count": 50,  "skip": 600, "replace": False},
    #{"name": "two spotted spider mite",      "count": 50,  "skip": 100, "replace": False},
    #{"name": "striped cucumber beetle",      "count": 400, "skip": 0,   "replace": False}, 
    {"name": "Strawberry Root Weevil",               "count": 50,  "skip": 650, "replace": False}
    #{"name": "Western Flower Thrips", "count": 400, "skip": 0, "replace": False, "quality_grade": "research,needs_id"}
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_species_folder(folder: Path) -> None:
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def get_extension_from_url(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext if ext in IMAGE_EXTS else ".jpg"


def normalize_photo_url(url: str, size: str = IMAGE_SIZE) -> str:
    return re.sub(
        r"/(square|small|medium|large|original)\.",
        f"/{size}.",
        url,
        flags=re.IGNORECASE,
    )


def fetch_observation_page(
    session: requests.Session,
    taxon_name: str,
    page: int,
    quality_grade: str = "research",
) -> dict:
    params = {
        "taxon_name": taxon_name,
        "photos": "true",
        "quality_grade": quality_grade,
        "per_page": PER_PAGE,
        "page": page,
        "order_by": "created_at",
        "order": "desc",
    }
    if PHOTO_LICENSES:
        params["photo_license"] = PHOTO_LICENSES

    resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def collect_photo_records(
    session: requests.Session,
    taxon_name: str,
    wanted_count: int,
    skip: int = 0,
    quality_grade: str = "research",
) -> tuple[List[Dict], int, int]:
    """
    Collect up to wanted_count unique photo records after skipping the first `skip`
    observation results conceptually via starting page and in-page skip.

    Returns:
        (records, pages_used, start_page)
    """
    seen_photo_ids: Set[int] = set()
    collected: List[Dict] = []

    start_page = skip // PER_PAGE + 1
    in_page_skip = skip % PER_PAGE

    page = start_page
    pages_used = 0
    first_page = True

    while len(collected) < wanted_count:
        data = fetch_observation_page(session, taxon_name, page, quality_grade)
        pages_used += 1

        results = data.get("results", [])
        if not results:
            break

        if first_page and in_page_skip > 0:
            results = results[in_page_skip:]
            first_page = False
        else:
            first_page = False

        for obs in results:
            obs_id = obs.get("id") or ""
            obs_uri = obs.get("uri") or (f"https://www.inaturalist.org/observations/{obs_id}" if obs_id else "")
            observed_on = obs.get("observed_on") or ""
            user_login = (obs.get("user") or {}).get("login") or ""
            user_name = (obs.get("user") or {}).get("name") or ""
            obs_license = obs.get("license_code") or ""

            for photo in obs.get("photos", []):
                photo_id = photo.get("id")
                raw_url = photo.get("url")
                if not photo_id or not raw_url:
                    continue
                if photo_id in seen_photo_ids:
                    continue

                seen_photo_ids.add(photo_id)

                collected.append({
                    "species_query": taxon_name,
                    "observation_id": obs_id,
                    "observation_url": obs_uri,
                    "observed_on": observed_on,
                    "user_login": user_login,
                    "user_name": user_name,
                    "observation_license_code": obs_license,
                    "photo_id": photo_id,
                    "image_url": normalize_photo_url(raw_url, IMAGE_SIZE),
                    "photo_attribution": photo.get("attribution") or "",
                    "photo_license_code": photo.get("license_code") or "",
                    "native_page_url": photo.get("native_page_url") or "",
                })

                if len(collected) >= wanted_count:
                    break

            if len(collected) >= wanted_count:
                break

        page += 1
        time.sleep(API_SLEEP_SECONDS)

    return collected, pages_used, start_page


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if "image" not in ctype.lower():
                return False

            with open(tmp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

        if not tmp_dest.exists() or tmp_dest.stat().st_size == 0:
            if tmp_dest.exists():
                tmp_dest.unlink()
            return False

        tmp_dest.replace(dest)
        return True

    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


def append_csv_rows(csv_path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "species_query",
        "species_slug",
        "requested_count",
        "skip",
        "quality_grade",
        "saved_filename",
        "saved_path",
        "photo_id",
        "observation_id",
        "observation_url",
        "image_url",
        "observed_on",
        "user_login",
        "user_name",
        "photo_attribution",
        "photo_license_code",
        "observation_license_code",
        "native_page_url",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_page_log(name: str, skip: int, start_page: int, pages_used: int, quality_grade: str) -> None:
    end_page = start_page + max(pages_used - 1, 0)
    with open(PAGE_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{name} | skip={skip} | quality_grade={quality_grade} | start_page={start_page} | "
            f"pages_used={pages_used} | end_page={end_page}\n"
        )


def existing_downloaded_count(folder: Path, slug: str) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS and f.stem.startswith(f"{slug}_")
    )


def run_species_job(
    session: requests.Session,
    name: str,
    wanted_count: int,
    skip: int,
    replace: bool,
    quality_grade: str = "research",
) -> None:
    slug = slugify(name)
    out_dir = OUTPUT_ROOT / slug
    ensure_dir(out_dir)

    if replace:
        print(f"\n[RESET] Clearing existing files for: {name}")
        clear_species_folder(out_dir)
        ensure_dir(out_dir)

    current = existing_downloaded_count(out_dir, slug)
    remaining = max(0, wanted_count - current)

    print(f"\n=== {name} ===")
    print(f"Folder        : {out_dir}")
    print(f"Target        : {wanted_count}")
    print(f"Current       : {current}")
    print(f"Need          : {remaining}")
    print(f"Skip          : {skip}")
    print(f"Quality grade : {quality_grade}")

    if remaining == 0:
        print("[SKIP] Already at target.")
        append_page_log(name, skip, start_page=(skip // PER_PAGE + 1), pages_used=0, quality_grade=quality_grade)
        return

    photo_records, pages_used, start_page = collect_photo_records(
        session=session,
        taxon_name=name,
        wanted_count=remaining,
        skip=skip,
        quality_grade=quality_grade,
    )

    append_page_log(name, skip, start_page, pages_used, quality_grade)

    if not photo_records:
        print("[WARN] No records found after skip/filtering.")
        return

    downloaded = 0
    failed = 0
    skipped_existing = 0
    csv_rows: List[Dict] = []

    for rec in photo_records:
        photo_id = rec["photo_id"]
        url = rec["image_url"]
        ext = get_extension_from_url(url)
        filename = f"{slug}_{photo_id}{ext}"
        dest = out_dir / filename

        if dest.exists():
            skipped_existing += 1
            continue

        ok = download_image(session, url, dest)
        if ok:
            downloaded += 1
            csv_rows.append({
                "species_query": rec["species_query"],
                "species_slug": slug,
                "requested_count": wanted_count,
                "skip": skip,
                "quality_grade": quality_grade,
                "saved_filename": filename,
                "saved_path": str(dest.resolve()),
                "photo_id": rec["photo_id"],
                "observation_id": rec["observation_id"],
                "observation_url": rec["observation_url"],
                "image_url": rec["image_url"],
                "observed_on": rec["observed_on"],
                "user_login": rec["user_login"],
                "user_name": rec["user_name"],
                "photo_attribution": rec["photo_attribution"],
                "photo_license_code": rec["photo_license_code"],
                "observation_license_code": rec["observation_license_code"],
                "native_page_url": rec["native_page_url"],
            })
            print(f"[OK]   {filename}")
        else:
            failed += 1
            print(f"[FAIL] {filename}")

        if current + downloaded >= wanted_count:
            break

        time.sleep(IMAGE_SLEEP_SECONDS)

    if csv_rows:
        append_csv_rows(REF_CSV, csv_rows)

    final_count = existing_downloaded_count(out_dir, slug)
    possible_downloads = len(photo_records)

    print(f"\n[SUMMARY] {name}")
    print(f"Downloaded new : {downloaded}")
    print(f"Skipped exist. : {skipped_existing}")
    print(f"Failed         : {failed}")
    print(f"Candidate recs : {possible_downloads}")
    print(f"Pages used     : {pages_used}")
    print(f"Final count    : {final_count}/{wanted_count}")

    if downloaded < remaining:
        print("[NOTE] Fewer images were available/downloadable than requested.")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "GrowLiv-iNat-downloader/1.0"
        })

        for item in SPECIES_CONFIG:
            run_species_job(
                session=session,
                name=item["name"],
                wanted_count=item["count"],
                skip=item.get("skip", 0),
                replace=item.get("replace", False),
                quality_grade=item.get("quality_grade", "research"),
            )

    print(f"\nCSV log written/appended to: {REF_CSV.resolve()}")
    print(f"Page log written/appended to: {PAGE_LOG.resolve()}")


if __name__ == "__main__":
    main()