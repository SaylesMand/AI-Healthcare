from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import requests

from resident.config import CR_ID, RAW_DIR

API = "https://apicr.minzdrav.gov.ru/api.ashx"


def fetch_clinrec(cr_id: str) -> dict:
    """Скачивает КР по id из API рубрикатора Минздрава."""
    resp = requests.get(
        API,
        params={"op": "GetClinrec2", "id": cr_id, "ssid": ""},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def save_snapshot(cr_id: str = CR_ID) -> dict:
    """Скачивает КР и сохраняет снимок с датой загрузки в data/raw."""
    payload = fetch_clinrec(cr_id)
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{API}?op=GetClinrec2&id={cr_id}",
        "clinrec": payload,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{cr_id}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    return snapshot


def load_snapshot(cr_id: str = CR_ID) -> dict:
    """Загружает сохраненный снимок КР из data/raw."""
    return json.loads((RAW_DIR / f"{cr_id}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=CR_ID)
    args = ap.parse_args()
    snap = save_snapshot(args.id)
    cr = snap["clinrec"]
    print(f"saved {args.id}: {cr['name']} (published {cr['publish_date'][:10]}, status {cr['status']})")
