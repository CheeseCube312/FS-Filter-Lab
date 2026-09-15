# FS FilterLab — Services Simplification Brief

Work from the repo root:
`C:\Users\JPC\Desktop\Foto\Programming\Filter Development\Dev folder\FS Filter Lab`
(Always. Running the app from any other folder creates junk directories like `program\program\`.)

## Mission

Shrink the `program/services/` layer: delete files, functions, and duplication.
**Zero behavior change.** Every chart, number, message, and file output must come
out identical. You are a deletion machine — if an edit adds more lines than it
removes, reconsider it.

## Hard rules (read twice)

1. **Do not touch:** `program/models/` (finished), `program/views/` (except the
   two import lines named in Phase 3), and `program/services/data.py` logic
   (Phase 3 adds one function to it — additive only, no rework).
2. **Do not touch the ECOSIS code** in `importing.py`: `import_ecosis_csv` and
   every helper only it uses. It is format-locked and golden-verified.
3. **Do not unify the two filename sanitizers.** `importing._sanitize_filename`
   and the report's `sanitize_filename_component` are deliberately different
   (documented in the code). Leave both.
4. Do not rename any public function or change any function signature except
   where this brief explicitly says "delete X".
5. Do not reformat, re-wrap, or "clean up" code you are not otherwise editing.
   Keep diffs minimal. No new comments narrating your changes.
6. Do not delete existing error handling, logging, or try/except blocks.
7. Do not touch `CACHE_FORMAT` or the cache pack/unpack logic in `data.py`.
8. Line numbers below are approximate; **symbol names are exact**. If a symbol
   this brief names does not exist, or code does not match the description,
   STOP that step, note it, and move on. Do not improvise.
9. Scale law: files store 0–100, internal arrays are 0–1. Never "fix" a `/ 100`
   or `* 100`.
10. A fresh behavior test suite now lives in `program/tests/` (stdlib
    unittest + numpy/pandas/plotly/matplotlib only — nothing new to install).
    It is green against the current code. Run it from the program/ directory:
    `python -m unittest discover -s tests`. Every phase must keep it green.
    If a phase changes behavior, the PHASE is wrong — fix the phase, not the
    test (the suite is behavior-only by design; it never pins signatures).

## Verification protocol (after EVERY phase)

From the repo root, in this order — all must pass before starting the next phase:

```powershell
# 1. Everything compiles
Get-ChildItem program -Recurse -Filter *.py | Where-Object FullName -notmatch __pycache__ | ForEach-Object { python -m py_compile $_.FullName }; echo COMPILE-OK

# 2. No dangling references to symbols deleted in this phase (edit the pattern)
Get-ChildItem program -Recurse -Filter *.py | Where-Object FullName -notmatch __pycache__ | Select-String -Pattern "SYMBOL_NAME"

# 3. Service import chain loads (remove modules you deleted)
python -c "import sys; sys.path.insert(0,'program'); import services.data, services.spectra, services.calculations, services.state_manager, services.visualization, services.presenter, services.importing; print('IMPORTS-OK')"

# 4. App boots: run `python program\app.py`, wait ~10 s, output must show no
#    traceback, then stop it with Ctrl+C.

# 5. Behavior suite green (the safety net for everything above)
Set-Location program; python -m unittest discover -s tests; Set-Location ..
```

If any step fails: fix it or revert the phase. One phase = one commit.

---

## Phase 0 — Baseline

Run the verification protocol once before changing anything (skip the
dangling-reference step), including the behavior suite (step 5). Confirm the
app boots. If anything fails before you start — including a single suite
test — stop and report; do not begin.

## Phase 1 — `services/state_manager.py` (safe warm-up)

1. **Delete the `show_channel_mixer` property AND its setter** (~lines 104–113,
   directly under the "COMPOSED-VIEW PROPERTIES" banner; delete the banner too).
   Verified: zero references anywhere else in the codebase. Re-verify with the
   grep step before deleting.
2. **Dedupe the two JSON persistence pairs** into two module-level helpers:
   `_load_json_list(path, key) -> list` and `_save_json_list(path, key, items)`
   (keep the `logger.warning` behavior and the parent-`mkdir` on save).
   Then:
   - `_load_default_reflectors_from_file` / `_save_default_reflectors_to_file`
     become one-liners around them (key `"default_reflectors"`,
     path `DEFAULT_REFLECTORS_FILE`).
   - `_load_my_filters_from_file` / `_save_my_filters_to_file` likewise
     (key `"my_filters"`, path `MY_FILTERS_FILE`).
   - **Preserve exactly:** My Filters saves with `sorted(filter_names)`;
     default reflectors save the list unsorted. Do not change JSON shapes.
3. **Hoist the function-local imports** (`from services.calculations import ...`
   inside `set_white_balance_from_surface` and `recompute_white_balance`) to the
   top of the module. There is no import cycle (calculations imports only
   models + services.spectra/channel_mixer). If step 3 of the protocol then
   reports a circular import, revert this item only and keep the local imports.

## Phase 2 — Delete `services/channel_mixer.py` (9 → 8 service files)

1. Move `apply_channel_mixing_to_responses` and `apply_channel_mixing_to_colors`
   into `services/spectra.py` (near `RGB_CHANNELS`), code unchanged — trim each
   docstring to one line. `ChannelMixerSettings` stays in `models/core.py`.
2. Delete `program/services/channel_mixer.py`.
3. Update imports: `services/calculations.py` (~line 19) and
   `services/visualization.py` (~line 26) now import the two functions from
   `services.spectra`.
4. Update the `services/__init__.py` docstring module list (drop channel_mixer).
5. Update `UX_SPEC.md`: remove `services.channel_mixer` from the
   forbidden-imports list (the line reading "views/ must NOT import
   services.calculations, services.visualization, ...").
6. Grep pattern for step 2: `channel_mixer`.

## Phase 3 — Delete `services/app_operations.py` (8 → 7 service files)

The file holds three unrelated things. Relocate, then delete the file.

1. `initialize_application_data()` and its helper `_try_operation()` move to the
   **bottom of `services/data.py`** (public function at the end). Code unchanged.
   `program/app.py` (~line 22) changes to
   `from services.data import initialize_application_data`.
2. `generate_application_report()` and `sanitize_filename_component()` move to
   **`services/visualization.py`**, next to `generate_report_png` (its only real
   consumer via the `sanitize_fn` argument). Add the imports visualization.py
   needs (`compute_selected_filter_indices`, `compute_filter_transmission`,
   `compute_effective_stops` from services.calculations; `AppState` from
   services.state_manager). Code unchanged — keep the
   "Mirror the on-screen numbers exactly" comment, it encodes a real contract.
   `program/views/actions.py` (~line 25) changes to
   `from services.visualization import generate_application_report`.
3. Delete `program/services/app_operations.py`. This also removes its misplaced
   mid-file import block (~line 66) automatically.
4. Update `services/__init__.py` docstring (drop app_operations).
5. Grep patterns: `app_operations`, `_try_operation`.
6. Watch for a circular import: visualization → calculations → spectra is fine;
   if the protocol's import step fails, report before proceeding.

## Phase 4 — `services/calculations.py` diet

1. **Delete `compute_active_transmission`.** It has exactly three users:
   - inside `effective_transmission` → inline its 3 lines there
     (gather rows for the indices, `combine_transmission(..., combine=True)`).
   - presenter film-emulation block (~line 205) → replace with
     `effective_transmission(emul_filter_names, {}, filter_collection)`
     (that call site already passes `{}` as multipliers — verified equivalent).
   - presenter `prepare_sensor_analysis` (~line 243) → replace with
     `effective_transmission(app_state.selected_filters, app_state.filter_multipliers, filter_collection)`
     — but FIRST check how that function computed `selected_indices`; if it used
     `app_state.filter_multipliers`, the swap is exact. If not, stop and report.
2. Keep `compute_filter_transmission` (indices → `(transmission, label,
   combined)` tuple) — presenter and the report depend on its exact return
   contract. You may simplify its internals only if behavior is identical.
3. **Move `format_transmission_metrics` and `format_white_balance_data` to
   `services/presenter.py`** — they are pure string formatters and presenter is
   their only consumer. Remove them from presenter's
   `from services.calculations import (...)` list.
4. Leave the white-balance wrappers, sRGB helpers, and
   `filter_display_color` alone this phase.
5. Grep pattern: `compute_active_transmission`.

## Phase 5 — `services/presenter.py` audit (read fully FIRST)

Read the whole file before editing. Then:
- If 3+ `prepare_*` functions repeat the same "resolve current_qe /
  illuminant / effective_transmission" preamble identically, factor ONE private
  helper (~5 lines). If fewer than 3 truly identical sites: do nothing.
- Do not move figure-building out of presenter (views receive pre-built
  figures via view models by design).
- Anything else you notice: note it in your report, do not act on it.

## Phase 6 — `services/visualization.py` audit (careful)

Read the whole file. Only dedupe clearly duplicated Plotly layout/trace styling.
Do NOT alter any numeric behavior: spectrum strips, saturation scaling,
peak-relative scaling, sRGB encoding, film-target scaling math are contracts.
When in doubt, leave it. Report what you found either way.

## Phase 7 — `services/importing.py` audit (most restricted)

Everything outside the ECOSIS block was unified recently; expect little. Only
remove something if you can grep-prove zero references. ECOSIS block: untouched,
no exceptions.

## Phase 8 — Docs and debris

1. Delete the stray directory `program\program\` and the file `boot_out.log`
   (repo root). Both are debris from running with the wrong working directory.
2. `TECHNICAL.md`: update ONLY the "Project Structure" tree so `services/`
   lists the seven remaining files (add `spectra.py` — it is currently missing;
   remove `channel_mixer.py` and `app_operations.py`). Do not rewrite the
   roadmap/progress-log sections — they are dated history.
3. UX_SPEC.md: already touched in Phase 2; re-check nothing else references
   deleted modules.

## Definition of done

- `program/services/` contains exactly: `__init__.py`, `data.py`,
  `importing.py`, `spectra.py`, `calculations.py`, `state_manager.py`,
  `visualization.py`, `presenter.py`.
- Full verification protocol green, including the behavior suite
  (`program/tests/`, run from `program/`).
- App boots from repo root, filter/sensor/surfaces/Film Match tabs render,
  a PNG report generates without error.
- Net line count across `program/services/` and touched files has gone DOWN.
  If it hasn't, you did it wrong — review your diffs for additions.
