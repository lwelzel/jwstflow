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


def test_every_observation_is_queried_on_its_own():
    """MAST answers a list of obs_id wildcards with the first one's observations only (HH 30's
    second MIRI epoch, observation 16, went missing that way): one query per observation, stacked."""
    from astropy.table import Table

    from jwstflow.data.download import query_observations

    class FakeMast:          # behaves like MAST: of a list of patterns only the first is used
        calls: list = []

        @classmethod
        def query_criteria(cls, **crit):
            cls.calls.append(crit["obs_id"])
            pattern = crit["obs_id"][0] if isinstance(crit["obs_id"], list) else crit["obs_id"]
            obs = pattern.split("-o")[1][:3]
            return Table({"obs_id": [f"jw02562-o{obs}_t004_miri_{f}" for f in ("f770w", "f1280w")]})

    cfg = DownloadConfig(program=2562, observations=[6, 16], instrument="MIRI", modes=["IMAGE"])
    table = query_observations(cfg, FakeMast)
    assert sorted(set(str(o)[:12] for o in table["obs_id"])) == ["jw02562-o006", "jw02562-o016"]
    assert len(table) == 4 and FakeMast.calls == ["jw02562-o006*", "jw02562-o016*"]
    single = query_observations(DownloadConfig(program=2562, observations=[6], instrument="MIRI"), FakeMast)
    assert len(single) == 2
