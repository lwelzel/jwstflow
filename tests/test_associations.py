"""Association building: per-exposure member rules, level-3 grouping, product names."""

from __future__ import annotations

import json
from pathlib import Path

from jwstflow.engine.associations import Association, build_associations, members_of, product_name
from jwstflow.config.schema import AssociationConfig
from jwstflow.data.discovery import FileRecord


def record(tmp_path: Path, name: str, **meta) -> FileRecord:
    p = tmp_path / name
    p.write_text("x")
    defaults = {"PROGRAM": "01751", "OBSERVTN": "010", "INSTRUME": "MIRI", "TARGPROP": "ESO-HA-569",
                "TARGID": "t005", "BKGDTARG": False, "IS_IMPRT": False}
    return FileRecord(p, {**defaults, **{k.upper(): v for k, v in meta.items()}})


def test_per_exposure_attaches_matching_backgrounds(tmp_path: Path):
    records = [
        record(tmp_path, "sci1_rate.fits", DETECTOR="MIRIFUSHORT", CHANNEL="12", BAND="SHORT"),
        record(tmp_path, "sci2_rate.fits", DETECTOR="MIRIFULONG", CHANNEL="34", BAND="SHORT"),
        record(tmp_path, "bkg1_rate.fits", DETECTOR="MIRIFUSHORT", CHANNEL="12", BAND="SHORT", BKGDTARG=True),
        record(tmp_path, "bkg2_rate.fits", DETECTOR="MIRIFUSHORT", CHANNEL="12", BAND="LONG", BKGDTARG=True),
    ]
    cfg = AssociationConfig(mode="per_exposure", level=2,
                            science_filters={"BKGDTARG": False},
                            members=[{"exptype": "background", "filters": {"BKGDTARG": True},
                                      "match_on": ["CHANNEL", "BAND"]}])
    asns = build_associations(records, cfg, asn_type="spec2", stage="calwebb_spec2")
    assert [a.name for a in asns] == ["sci1_spec2", "sci2_spec2"]  # backgrounds are never science
    by_name = {a.name: a.data["products"][0]["members"] for a in asns}
    assert [(Path(m["expname"]).name, m["exptype"]) for m in by_name["sci1_spec2"]] == [
        ("sci1_rate.fits", "science"), ("bkg1_rate.fits", "background")]  # band-matched only
    assert [m["exptype"] for m in by_name["sci2_spec2"]] == ["science"]  # no MIRIFULONG background


def test_member_rules_differ_on(tmp_path: Path):
    records = [
        record(tmp_path, "nod1_rate.fits", DETECTOR="NRS1", PATT_NUM=1, INSTRUME="NIRSPEC"),
        record(tmp_path, "nod2_rate.fits", DETECTOR="NRS1", PATT_NUM=2, INSTRUME="NIRSPEC"),
    ]
    cfg = AssociationConfig(mode="per_exposure", level=2,
                            members=[{"exptype": "background", "filters": {},
                                      "match_on": ["DETECTOR"], "differ_on": ["PATT_NUM"]}])
    asns = build_associations(records, cfg, asn_type="spec2", stage="s")
    members = asns[0].data["products"][0]["members"]
    assert [(Path(m["expname"]).name, m["exptype"]) for m in members] == [
        ("nod1_rate.fits", "science"), ("nod2_rate.fits", "background")]  # the *other* nod


def test_grouped_associations_and_dms_product_names(tmp_path: Path):
    records = [
        record(tmp_path, f"exp{i}_{g.lower()}_cal.fits", INSTRUME="NIRSPEC", GRATING=g, FILTER=f, PATT_NUM=i)
        for g, f in (("G235H", "F170LP"), ("G395H", "F290LP")) for i in (1, 2)
    ]
    cfg = AssociationConfig(mode="group", level=3, group_by=["PROGRAM", "OBSERVTN", "GRATING", "FILTER"],
                            product_name="jw{PROGRAM}-o{OBSERVTN}_{TARGID}_nirspec_{GRATING}")
    asns = build_associations(records, cfg, asn_type="spec3", stage="calwebb_spec3")
    assert sorted(a.data["products"][0]["name"] for a in asns) == [
        "jw01751-o010_t005_nirspec_g235h", "jw01751-o010_t005_nirspec_g395h"]
    assert all(len(a.data["products"][0]["members"]) == 2 for a in asns)
    assert all(a.data["asn_id"] == "o010" and a.data["asn_type"] == "spec3" for a in asns)


def test_grouped_labels_deduplicate_when_product_names_collide(tmp_path: Path):
    # DMS-style IFU names without the grating: cube_build appends it, so two groups
    # share one product name -- the association files must still be distinct
    records = [
        record(tmp_path, f"e{i}_{g.lower()}_cal.fits", INSTRUME="NIRSPEC", GRATING=g, FILTER=f)
        for i, (g, f) in enumerate([("G235H", "F170LP"), ("G395H", "F290LP")])
    ]
    cfg = AssociationConfig(mode="group", level=3, group_by=["GRATING", "FILTER"],
                            product_name="jw{PROGRAM}-o{OBSERVTN}_{TARGID}_nirspec")
    asns = build_associations(records, cfg, asn_type="spec3", stage="s")
    assert len({a.filename for a in asns}) == 2
    assert len({a.data["products"][0]["name"] for a in asns}) == 1


def test_product_name_pads_and_falls_back_to_targprop(tmp_path: Path):
    rec = FileRecord(tmp_path / "x.fits", {"PROGRAM": "1751", "OBSERVTN": "6", "TARGPROP": "ESO-Ha 569",
                                           "INSTRUME": "NIRSPEC", "GRATING": "G235H"})
    name = product_name("jw{PROGRAM}-o{OBSERVTN}_{TARGID}_{INSTRUME}_{GRATING}", rec)
    assert name == "jw01751-o006_eso-ha-569_nirspec_g235h"  # zero-padded, TARGID from TARGPROP
    rec.meta["TARGID"] = "t005"
    assert "t005" in product_name("{TARGID}", rec)


def test_default_science_filters_exclude_backgrounds_at_level3(tmp_path: Path):
    records = [record(tmp_path, "sci_cal.fits", GRATING="G235H"),
               record(tmp_path, "bkg_cal.fits", GRATING="G235H", BKGDTARG=True)]
    cfg = AssociationConfig(mode="group", level=3, group_by=["GRATING"])
    asns = build_associations(records, cfg, asn_type="spec3", stage="s")
    members = [Path(m["expname"]).name for a in asns for m in a.data["products"][0]["members"]]
    assert members == ["sci_cal.fits"]


def test_write_and_members_roundtrip(tmp_path: Path):
    member = tmp_path / "m_rate.fits"
    member.write_text("x")
    asn = Association("demo_spec2", {
        "asn_type": "spec2", "products": [{"name": "demo", "members": [
            {"expname": str(member), "exptype": "science"}]}]}, [member])
    path = asn.write(tmp_path / "asn")
    assert path.name == "demo_spec2_asn.json"
    assert members_of(path) == [member]
    relative = asn.write(tmp_path / "asn_rel", relative=True)
    data = json.loads(relative.read_text())
    assert not Path(data["products"][0]["members"][0]["expname"]).is_absolute()
    assert members_of(relative) == [tmp_path / "asn_rel" / ".." / "m_rate.fits"]


def test_optelem_follows_the_dms_rule_for_imagers(tmp_path: Path):
    """{OPTELEM} of imaging products is MAST's: filter and pupil wheel, empty ones dropped, sorted
    case-blind (jwst dms_base.get_opt_element) -- spectroscopic modes keep their own forms."""
    def name(**keys):
        rec = FileRecord(tmp_path / "x.fits", {"PROGRAM": "2562", "OBSERVTN": "1", "TARGID": "t001", **keys})
        return product_name("jw{PROGRAM}-o{OBSERVTN}_{TARGID}_{INSTRUME}_{OPTELEM}", rec)

    assert name(INSTRUME="NIRCAM", FILTER="F200W", PUPIL="CLEAR") == "jw02562-o001_t001_nircam_clear-f200w"
    assert name(INSTRUME="NIRCAM", FILTER="F444W", PUPIL="F405N") == "jw02562-o001_t001_nircam_f405n-f444w"
    assert name(INSTRUME="NIRISS", FILTER="CLEAR", PUPIL="F150W") == "jw02562-o001_t001_niriss_clear-f150w"
    assert name(INSTRUME="MIRI", FILTER="F770W") == "jw02562-o001_t001_miri_f770w"
    assert name(INSTRUME="MIRI", FILTER="F1280W", PUPIL="N/A") == "jw02562-o001_t001_miri_f1280w"
    assert name(INSTRUME="MIRI", CHANNEL="12", BAND="SHORT") == "jw02562-o001_t001_miri_12-short"
    assert name(INSTRUME="NIRSPEC", GRATING="G395H", FILTER="F290LP") == "jw02562-o001_t001_nirspec_g395h-f290lp"


def test_subarray_suffix_follows_the_dms_rule(tmp_path: Path):
    """DMS appends the subarray, in lower case, unless it is the full array (MAST's
    jw04290-o004_t002_miri_f770w-brightsky_i2d.fits)."""
    template = "jw{PROGRAM}-o{OBSERVTN}_{TARGID}_{INSTRUME}_{OPTELEM}{SUBARRAY_SUFFIX}"
    def name(**keys):
        rec = FileRecord(tmp_path / "x.fits", {"PROGRAM": "4290", "OBSERVTN": "4", "TARGID": "t002", **keys})
        return product_name(template, rec)

    assert name(INSTRUME="MIRI", FILTER="F770W", SUBARRAY="BRIGHTSKY") == "jw04290-o004_t002_miri_f770w-brightsky"
    assert name(INSTRUME="MIRI", FILTER="F770W", SUBARRAY="FULL") == "jw04290-o004_t002_miri_f770w"
    assert name(INSTRUME="NIRCAM", FILTER="F444W", PUPIL="CLEAR") == "jw04290-o004_t002_nircam_clear-f444w"
    assert name(INSTRUME="NIRCAM", FILTER="F200W", PUPIL="CLEAR", SUBARRAY="SUB640") == \
        "jw04290-o004_t002_nircam_clear-f200w-sub640"
