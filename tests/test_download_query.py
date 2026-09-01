"""The MAST query construction (network-free part of jwstflow.data.download)."""

from __future__ import annotations

from jwstflow.config.schema import DownloadConfig
from jwstflow.data.download import build_query


def test_build_query_expands_program_observations_and_modes():
    cfg = DownloadConfig(program=1751, observations=[6, 10], instrument="NIRSPEC", modes=["IFU"])
    crit = build_query(cfg)
    assert crit["obs_collection"] == "JWST"
    assert crit["proposal_id"] == ["01751", "1751"]  # MAST is inconsistent about zero padding
    assert crit["obs_id"] == ["jw01751-o006*", "jw01751-o010*"]
    assert all("NIRSPEC/IFU" in n or "NIRSPEC" in n for n in crit["instrument_name"])


def test_build_query_extras_pass_through():
    cfg = DownloadConfig(program=1, instrument="MIRI", exclusive_only=True,
                         query_extra={"t_exptime": [100, 200]})
    crit = build_query(cfg)
    assert crit["dataRights"] == "EXCLUSIVE_ACCESS"
    assert crit["t_exptime"] == [100, 200]
