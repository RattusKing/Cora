import math

from cora import anage, config

HEADER = (
    "HAGRID\tKingdom\tPhylum\tClass\tOrder\tFamily\tGenus\tSpecies\tCommon name\tFemale maturity (days)\tMale maturity (days)"
    "\tGestation/Incubation (days)\tWeaning (days)\tLitter/Clutch size\tLitters/Clutches per year\tInter-litter/Interbirth interval"
    "\tBirth weight (g)\tWeaning weight (g)\tAdult weight (g)\tGrowth rate (1/days)\tMaximum longevity (yrs)\tSource\tSpecimen origin"
    "\tSample size\tData quality\tIMR (per yr)\tMRDT (yrs)\tMetabolic rate (W)\tBody mass (g)\tTemperature (K)\tReferences"
)


def _row(hagrid, cls, genus, species, common, longevity, quality="acceptable", size="medium", origin="wild", adult_w="", mass="", temp="", metab=""):
    cells = [hagrid, "Animalia", "Chordata", cls, "Ord", "Fam", genus, species, common, "", "", "", "", "", "", "", "", "", adult_w, "", str(longevity), "1", origin, size, quality, "", "", metab, mass, temp, "1"]
    return "\t".join(cells)


def _mammal_rows():
    """12 synthetic mammals exactly on the line longevity = 2 * mass^0.25 (log-log linear)."""
    rows = []
    for i in range(12):
        mass = 10 ** (i * 0.5 + 1)  # 10 g .. 10^6.5 g
        lon = 2 * mass ** 0.25
        rows.append(_row(f"M{i:03d}", "Mammalia", f"Genus{i}", f"sp{i}", f"mammal {i}", f"{lon:.4f}", mass=f"{mass:.2f}"))
    return rows


def _fixture():
    lines = [HEADER] + _mammal_rows() + [
        # an outlier with "questionable" quality: kept as a record, excluded from the fit
        _row("N001", "Mammalia", "Heterocephalus", "glaber", "Naked mole-rat", "37", quality="questionable", size="large", origin="captivity", mass="35", temp="305", metab="0.05"),
        _row("Q001", "Bivalvia", "Arctica", "islandica", "Ocean quahog", "507", quality="acceptable", size="small", origin="wild"),
        _row("R001", "Actinopterygii", "Sebastes", "aleutianus", "Rougheye rockfish", "205", adult_w="6000"),
        _row("R002", "Actinopterygii", "Sebastes", "mystinus", "Blue rockfish", "44", adult_w="2000"),
        _row("R003", "Actinopterygii", "Sebastes", "nothing", "No longevity", "", adult_w="100"),
        _row("X001", "Mammalia", "Bogus", "bad", "Unparseable", "abc", mass="oops"),
    ]
    return "\n".join(lines) + "\n"


def test_parse_maps_columns_and_types():
    rows = anage.parse(_fixture())
    by = {r["hagrid"]: r for r in rows}
    nmr = by["N001"]
    assert nmr["binomial"] == "Heterocephalus glaber" and nmr["class_"] == "Mammalia"
    assert nmr["max_longevity_yrs"] == 37.0 and nmr["body_mass_g"] == 35.0 and nmr["temperature_k"] == 305.0 and nmr["metabolic_rate_w"] == 0.05
    assert nmr["data_quality"] == "questionable" and nmr["sample_size"] == "large" and nmr["specimen_origin"] == "captivity"
    assert by["R003"]["max_longevity_yrs"] is None
    assert by["X001"]["max_longevity_yrs"] is None and by["X001"]["body_mass_g"] is None  # bad numbers -> None, row kept


def test_load_lookup_and_genus_match(conn):
    n = anage.load(conn, rows=anage.parse(_fixture()))
    assert n == anage.loaded_count(conn) == 18
    assert anage.lookup_binomial(conn, "Arctica islandica")["max_longevity_yrs"] == 507.0
    assert anage.lookup_binomial(conn, "Nope nope") is None
    traits = anage.panel_traits(conn)
    assert traits["naked_mole_rat"]["match"] == "exact"
    assert traits["ocean_quahog"]["hagrid"] == "Q001"
    # genus-level entry takes the longest-lived species with a longevity value
    assert traits["rockfish"]["species"] == "aleutianus" and traits["rockfish"]["match"] == "genus: longest-lived of 2 Sebastes species"
    assert traits["greenland_shark"] is None  # not in the fixture


def test_allometric_fit_recovers_the_synthetic_law_and_respects_quality(conn):
    anage.load(conn, rows=anage.parse(_fixture()))
    fit = anage.class_fit(conn, "Mammalia")
    # 12 synthetic rows fit; the questionable NMR row and the unparseable row are excluded
    assert fit is not None and fit["n"] == 12
    assert abs(fit["b"] - 0.25) < 1e-3 and abs(fit["a"] - math.log10(2)) < 1e-3
    assert anage.class_fit(conn, "Bivalvia") is None  # too few rows


def test_longevity_quotient(conn):
    anage.load(conn, rows=anage.parse(_fixture()))
    on_the_line = {"class_": "Mammalia", "max_longevity_yrs": 2 * (1000 ** 0.25), "body_mass_g": 1000.0}
    lq = anage.longevity_quotient(conn, on_the_line)
    assert lq is not None and abs(lq["lq"] - 1.0) < 0.01 and lq["fit_class"] == "Mammalia" and lq["fit_n"] == 12 and "naive" in lq["note"]
    twice = dict(on_the_line, max_longevity_yrs=on_the_line["max_longevity_yrs"] * 2)
    assert abs(anage.longevity_quotient(conn, twice)["lq"] - 2.0) < 0.02
    # the real naked mole-rat row: 37 yrs at 35 g against the synthetic law (2 * 35^0.25 = 4.86) -> LQ ~7.6
    nmr = anage.lookup_binomial(conn, "Heterocephalus glaber")
    assert abs(anage.longevity_quotient(conn, nmr)["lq"] - 37 / (2 * 35 ** 0.25)) < 0.05
    assert anage.longevity_quotient(conn, {"class_": "Mammalia", "max_longevity_yrs": 10}) is None  # no mass
    assert anage.longevity_quotient(conn, {"class_": "Bivalvia", "max_longevity_yrs": 507, "adult_weight_g": 100}) is None  # no fit
    assert anage.longevity_quotient(conn, None) is None


def test_panel_summary_and_render(conn):
    anage.load(conn, rows=anage.parse(_fixture()))
    rows = anage.panel_summary(conn)
    assert [r["key"] for r in rows] == list(config.SPECIES)
    nmr = next(r for r in rows if r["key"] == "naked_mole_rat")
    assert nmr["docs"] == 2 and nmr["anage"]["max_longevity_yrs"] == 37.0 and nmr["lq"] is not None
    shark = next(r for r in rows if r["key"] == "greenland_shark")
    assert shark["anage"] is None and shark["lq"] is None
    text = anage.render_summary(rows)
    assert "naked_mole_rat" in text and "no AnAge record" in text and "naive" in text


def test_full_panel_is_the_default():
    assert config.DEFAULT_SPECIES == list(config.SPECIES) and len(config.DEFAULT_SPECIES) == 8
    assert config.P05_SPECIES == ["naked_mole_rat", "ocean_quahog", "rockfish"]
