"""Input discovery (filters, cache) and the product-naming grammar."""

from __future__ import annotations

from pathlib import Path

import pytest

from jwstflow.data.discovery import FileRecord, HeaderCache, discover, match_filters
from jwstflow.naming import check_custom_suffix, derived_name, is_official_product, split_suffix
from jwstflow.testing import synthetic_image


# ------------------------------------------------------------------ filters


def test_match_filters_semantics():
    meta = {"EXP_TYPE": "NRS_IFU", "GRATING": "G235H", "PATT_NUM": 2}
    assert match_filters(meta, {"EXP_TYPE": "nrs_ifu"})          # case-insensitive strings
    assert match_filters(meta, {"EXP_TYPE": ["MIR_MRS", "NRS_IFU"]})  # list = membership
    assert match_filters(meta, {"GRATING": "regex:G2.5H"})       # regex: prefix
    assert not match_filters(meta, {"GRATING": "regex:^X"})
    assert match_filters(meta, {"PATT_NUM": 2}) and not match_filters(meta, {"PATT_NUM": 1})
    assert not match_filters(meta, {"NO_SUCH": "x"})             # missing keyword never matches ...
    assert match_filters(meta, {"NO_SUCH": None})                # ... except a null filter
    # a missing boolean keyword counts as False (BKGDTARG is absent on many products)
    assert match_filters(meta, {"BKGDTARG": False})
    assert not match_filters(meta, {"BKGDTARG": True})
    assert match_filters({"BKGDTARG": "T"}, {"BKGDTARG": True})  # FITS logical as string


def test_discover_filters_excludes_and_skips_tmp(tmp_path: Path):
    synthetic_image(tmp_path / "a_nrs1_rate.fits", EXP_TYPE="NRS_IFU")
    synthetic_image(tmp_path / "b_mrs_rate.fits", instrument="MIRI", EXP_TYPE="MIR_MRS")
    synthetic_image(tmp_path / "c_skipme_rate.fits", EXP_TYPE="NRS_IFU")
    tmp_task = tmp_path / ".tmp-abc"
    tmp_task.mkdir()
    synthetic_image(tmp_task / "half_written_rate.fits")
    found = discover(tmp_path, "*_rate.fits", recursive=True, exclude=["*skipme*"],
                     filters={"EXP_TYPE": "NRS_IFU"})
    assert [r.name for r in found] == ["a_nrs1_rate.fits"]
    assert found[0].get("DETECTOR") == "NRS1"
    assert discover(tmp_path / "missing", "*.fits") == []


def test_header_cache_tracks_file_identity(tmp_path: Path):
    img = synthetic_image(tmp_path / "a_rate.fits", EXP_TYPE="NRS_IFU")
    cache_file = tmp_path / "headers.json"
    cache = HeaderCache(cache_file)
    assert cache.get(img)["EXP_TYPE"] == "NRS_IFU"
    cache.save()
    assert cache_file.exists()
    # same size+mtime: answered from the cache (poison the entry to prove it)
    cache2 = HeaderCache(cache_file)
    key = cache2._key(img)
    cache2._data[key]["EXP_TYPE"] = "POISONED"
    assert cache2.get(img)["EXP_TYPE"] == "POISONED"
    # rewriting the file invalidates the key
    synthetic_image(img, EXP_TYPE="MIR_MRS", instrument="MIRI", shape=(65, 64))
    assert cache2.get(img)["EXP_TYPE"] == "MIR_MRS"


def test_association_files_are_summarised(tmp_path: Path):
    (tmp_path / "x_asn.json").write_text(
        '{"asn_type": "spec3", "products": [{"name": "p", "members": [{"expname": "a_cal.fits"}]}]}')
    (rec,) = discover(tmp_path, "*_asn.json")
    assert rec.get("KIND") == "asn" and rec.get("ASN_TYPE") == "spec3"


def test_file_record_stem_strips_the_product_suffix():
    assert FileRecord(Path("jw01_nrs1_uncal.fits")).stem == "jw01_nrs1"
    assert FileRecord(Path("jw01_nrs1_cal.fits")).stem == "jw01_nrs1"
    assert FileRecord(Path("x_asn.json")).stem == "x"


# ------------------------------------------------------------------ naming


def test_split_suffix_only_splits_known_products():
    assert split_suffix("jw01_nrs1_cal") == ("jw01_nrs1", "cal")
    assert split_suffix("jw01_nirspec_s1d") == ("jw01_nirspec", "s1d")
    assert split_suffix("jw01_g235h-f170lp") == ("jw01_g235h-f170lp", None)  # optelem is no suffix
    assert split_suffix("plainname") == ("plainname", None)


def test_derived_name_strips_inserts_and_appends():
    src = "jw01751-o010_t005_miri_ch1-short_s3d.fits"
    assert derived_name(src, "s1d", descriptor="eso-ha-569_circle1") == \
        "jw01751-o010_t005_miri_ch1-short_eso-ha-569_circle1_s1d.fits"
    assert derived_name("jw01_nrs1_cal.fits", "diskmask") == "jw01_nrs1_diskmask.fits"
    assert derived_name("jw01_nrs1_cal.fits", "s1dcomb", ext=".ecsv") == "jw01_nrs1_s1dcomb.ecsv"


def test_official_suffixes_are_reserved():
    with pytest.raises(ValueError, match="reserved"):
        check_custom_suffix("cal")
    with pytest.raises(ValueError, match="lowercase"):
        check_custom_suffix("Bad-Suffix")
    assert is_official_product("jw01_nrs1_s3d.fits")
    assert not is_official_product("jw01_nrs1_diskmask.fits")
