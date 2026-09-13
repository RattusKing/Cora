"""AnAge (HAGR) lifespan records for the species panel.

AnAge is CC BY 3.0. We download the dataset zip, keep the raw `anage_data.txt` locally
(git-ignored under data/), and load every row into `species_traits`: maximum longevity with
its **data quality, sample size and specimen origin** (the attributes the red-team asked
for), plus the covariates a residual-longevity phenotype needs later (body mass, metabolic
rate, temperature).

`longevity_quotient` is a deliberately *naive* first step toward residual longevity: a
class-level log-log fit of maximum longevity on body mass across AnAge, with no
phylogenetic correction. It is labeled as such wherever it is shown.
"""

from __future__ import annotations

import csv
import io
import math
import urllib.request
import zipfile
from pathlib import Path

from . import config, db

COLUMNS = {
    "HAGRID": "hagrid",
    "Class": "class_",
    "Order": "order_",
    "Family": "family",
    "Genus": "genus",
    "Species": "species",
    "Common name": "common_name",
    "Maximum longevity (yrs)": "max_longevity_yrs",
    "Data quality": "data_quality",
    "Sample size": "sample_size",
    "Specimen origin": "specimen_origin",
    "Adult weight (g)": "adult_weight_g",
    "Body mass (g)": "body_mass_g",
    "Metabolic rate (W)": "metabolic_rate_w",
    "Temperature (K)": "temperature_k",
    "Female maturity (days)": "female_maturity_days",
    "Growth rate (1/days)": "growth_rate",
    "Source": "source",
}
FLOAT_COLS = {"max_longevity_yrs", "adult_weight_g", "body_mass_g", "metabolic_rate_w", "temperature_k", "female_maturity_days", "growth_rate"}
FIT_QUALITY = ("acceptable", "high")
MIN_FIT_ROWS = 10
LQ_NOTE = "naive class-level allometric fit of max longevity on body mass; no phylogenetic correction"


def download(url: str | None = None, dest: Path | None = None) -> Path:
    """Fetch the AnAge zip and extract anage_data.txt to `dest`."""
    url = url or config.ANAGE_URL
    dest = dest or config.ANAGE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed HAGR host
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        dest.write_bytes(z.read("anage_data.txt"))
    return dest


def parse(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows: list[dict] = []
    for raw in reader:
        row: dict = {}
        for src, dst in COLUMNS.items():
            val = (raw.get(src) or "").strip()
            if dst in FLOAT_COLS:
                try:
                    row[dst] = float(val) if val else None
                except ValueError:
                    row[dst] = None
            else:
                row[dst] = val or None
        if not row.get("hagrid"):
            continue
        row["binomial"] = f"{row.get('genus') or ''} {row.get('species') or ''}".strip()
        rows.append(row)
    return rows


def load(conn, path: Path | None = None, rows: list[dict] | None = None) -> int:
    """Replace the species_traits table with the given rows (or the file at `path`)."""
    if rows is None:
        path = path or config.ANAGE_PATH
        rows = parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    conn.execute("DELETE FROM species_traits")
    now = db.now_iso()
    conn.executemany(
        """INSERT INTO species_traits (hagrid, class_, order_, family, genus, species, binomial, common_name,
             max_longevity_yrs, data_quality, sample_size, specimen_origin, adult_weight_g, body_mass_g,
             metabolic_rate_w, temperature_k, female_maturity_days, growth_rate, source, loaded_at)
           VALUES (:hagrid, :class_, :order_, :family, :genus, :species, :binomial, :common_name,
             :max_longevity_yrs, :data_quality, :sample_size, :specimen_origin, :adult_weight_g, :body_mass_g,
             :metabolic_rate_w, :temperature_k, :female_maturity_days, :growth_rate, :source, :loaded_at)""",
        [dict(r, loaded_at=now) for r in rows],
    )
    conn.commit()
    db.log_event(conn, "anage", {"rows": len(rows), "path": str(path) if path else None})
    return len(rows)


def loaded_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM species_traits").fetchone()["n"]


def lookup_binomial(conn, binomial: str) -> dict | None:
    row = conn.execute("SELECT * FROM species_traits WHERE binomial = ? LIMIT 1", (binomial,)).fetchone()
    return dict(row) if row else None


def genus_rows(conn, genus: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM species_traits WHERE genus = ? ORDER BY max_longevity_yrs DESC", (genus,)).fetchall()
    return [dict(r) for r in rows]


def panel_traits(conn) -> dict[str, dict | None]:
    """AnAge row per configured species. A genus-level panel entry (e.g. Sebastes) takes the
    longest-lived species in the genus and records how many candidates there were."""
    out: dict[str, dict | None] = {}
    for key, sp in config.SPECIES.items():
        if " " in sp.binomial:
            row = lookup_binomial(conn, sp.binomial)
            if row:
                row["match"] = "exact"
        else:
            rows = [r for r in genus_rows(conn, sp.binomial) if r.get("max_longevity_yrs")]
            row = rows[0] if rows else None
            if row:
                row["match"] = f"genus: longest-lived of {len(rows)} {sp.binomial} species"
        out[key] = row
    return out


def allometric_fit(rows: list[dict]) -> dict | None:
    """Least-squares fit of log10(max longevity) on log10(body mass). Returns None if too few rows."""
    pts = []
    for r in rows:
        mass = r.get("body_mass_g") or r.get("adult_weight_g")
        lon = r.get("max_longevity_yrs")
        if mass and lon and mass > 0 and lon > 0 and (r.get("data_quality") in FIT_QUALITY):
            pts.append((math.log10(mass), math.log10(lon)))
    n = len(pts)
    if n < MIN_FIT_ROWS:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return {"a": a, "b": b, "n": n}


def class_fit(conn, class_name: str) -> dict | None:
    rows = [dict(r) for r in conn.execute("SELECT * FROM species_traits WHERE class_ = ?", (class_name,)).fetchall()]
    return allometric_fit(rows)


def longevity_quotient(conn, trait: dict | None) -> dict | None:
    """Observed / predicted maximum longevity for one AnAge row, from its class's fit."""
    if not trait or not trait.get("max_longevity_yrs") or not trait.get("class_"):
        return None
    mass = trait.get("body_mass_g") or trait.get("adult_weight_g")
    if not mass or mass <= 0:
        return None
    fit = class_fit(conn, trait["class_"])
    if fit is None:
        return None
    predicted = 10 ** (fit["a"] + fit["b"] * math.log10(mass))
    return {
        "lq": round(trait["max_longevity_yrs"] / predicted, 2),
        "predicted_yrs": round(predicted, 1),
        "mass_g": mass,
        "fit_class": trait["class_"],
        "fit_n": fit["n"],
        "note": LQ_NOTE,
    }


def panel_summary(conn) -> list[dict]:
    """One row per configured species: docs in the corpus, AnAge attributes, naive LQ."""
    counts = db.count_docs_by_species(conn)
    traits = panel_traits(conn)
    out = []
    for key, sp in config.SPECIES.items():
        t = traits.get(key)
        out.append(
            {
                "key": key,
                "name": sp.name,
                "binomial": sp.binomial,
                "default": key in config.DEFAULT_SPECIES,
                "docs": counts.get(key, 0),
                "anage": None if not t else {
                    "binomial": t["binomial"], "match": t.get("match"), "hagrid": t["hagrid"], "class": t["class_"],
                    "max_longevity_yrs": t["max_longevity_yrs"], "data_quality": t["data_quality"],
                    "sample_size": t["sample_size"], "specimen_origin": t["specimen_origin"],
                    "body_mass_g": t.get("body_mass_g") or t.get("adult_weight_g"),
                    "metabolic_rate_w": t["metabolic_rate_w"], "temperature_k": t["temperature_k"],
                },
                "lq": longevity_quotient(conn, t),
            }
        )
    return out


def render_summary(rows: list[dict]) -> str:
    lines = [f"{'species':<18}{'docs':>5}  {'AnAge max (yrs)':<16}{'quality':<13}{'n':<8}{'origin':<10}{'mass (g)':>11}  {'LQ':>5}  match"]
    for r in rows:
        a = r["anage"]
        if a is None:
            lines.append(f"{r['key']:<18}{r['docs']:>5}  {'- no AnAge record -':<16}")
            continue
        lq = r["lq"]["lq"] if r["lq"] else None
        mass = a["body_mass_g"]
        lines.append(
            f"{r['key']:<18}{r['docs']:>5}  {a['max_longevity_yrs'] or '?':<16}{(a['data_quality'] or '?'):<13}{(a['sample_size'] or '?'):<8}{(a['specimen_origin'] or '?'):<10}"
            f"{(f'{mass:,.0f}' if mass else '?'):>11}  {(f'{lq:.2f}' if lq is not None else '  -'):>5}  {a['match']}"
        )
    lines.append(f"LQ = {LQ_NOTE}")
    return "\n".join(lines)
