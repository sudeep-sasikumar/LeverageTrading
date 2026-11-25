# Recovered research scripts (DESIGN.md sections 15–27)

These are the scripts behind the year studies of 16–20 Nov 2026, recovered verbatim from the Claude Code
session transcript that wrote them (each file is the last Write of that file plus every later Edit, replayed
in order). They were run from a temporary scratchpad that has since been cleared, so their `OUT = ...`
paths point to a folder that no longer exists, and the data they read (`aim_events.parquet`, `paths/`, `pre/`
chunks) is rebuilt by `research/build_events.py` and `research/fetch_year_5m.py` into `research_data/`.

They are kept as the record of what was run, not as maintained code. The maintained reproductions are
`research/reproduce_r3.py` (§25) and `research/reproduce_runners.py` (§27).

Map: aim runner = `aim_test.py` (§15), minute shape = `shape_study.py` (§16), tight stops / plans =
`tight_stops.py` (§17–18), pullback = `pullback.py` (§20), 5x factors/models = `five_x.py` (§21), levels =
`levels.py` (§23), ten-rule test = `quality.py` (§25), early exits = `exits.py`, `exits_flow.py` (§26),
runners = `runners.py`, `runners2.py`, `runners_1m.py` (§27), patterns = `run_pattern_study.py`,
`sweep_grid.py`, `sweep_control.py` (§14), path downloads = `fetch_paths.py`, `fetch_pre.py`, `fetch_flow.py`,
`fetch_1m.py`. `patch_*.py` are the code edits applied to the repository at the time.
