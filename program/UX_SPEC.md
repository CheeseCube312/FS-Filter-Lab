# FS Filter Lab — UI/UX Specification

**Status:** Authoritative build contract for the UI rebuild (`ui-rebuild` branch).
**Supersedes:** `USAGE.md` (rewritten from the new UI in U6) and the deleted `views/` layer.
**Backend policy:** `services/` and `models/` are tested and remain unchanged except §7
(computation changes), each of which lands with new tests. The unittest suite
(currently 95 tests) must stay green after every slice.

---

## 1. Product identity

- **Single-user, local, download-and-run.** No hosting, no accounts, no onboarding,
  no multi-user concessions. Every screen must be usable on first launch.
- **Stack:** Python + NiceGUI (Quasar/Vue) + Plotly. Backend stays
  framework-agnostic (no NiceGUI imports outside `views/` and `app.py`).
- **The shared spine:** every instrument derives from the triple
  **filter stack × sensor/film × illuminant**. All panels read the same
  `AppState`; editing any leg of the triple refreshes every instrument that
  depends on it (§5 refresh contract).

## 2. The three modes (from usage analysis)

| Mode | What it is | Frequency weight |
|---|---|---|
| **A — Quick answer** | Cold start → search/import → read charts → export report → share | constant background use |
| **B — Stack design** | Iterate a stack toward a visual target (e.g. "blue leaves under WB on gravel"), owned filters only | the flagship loop |
| **C — Film matching** | Load film spectral response, shape sensor RGB response to match, fine-tune with channel mixer | the headline new capability |

### Mode A — Quick answer
Walkthrough: search two filters → one missing → import digitized plot
(WebPlotDigitizer CSV) as a new filter → stack them → read combined
transmission + white-balanced RGB response ("the UV leak shows as yellow, the
IR leak pollutes blue") → export report → send it to someone.

Requirements:
- Search-first shelf; a chart can exist within a minute of cold start.
- Digitized-CSV import is a ≤2-click path at the top of the Data shelf.
- Report export lives in the header toolbar, always one click away.
- No configuration required before the first chart renders.

### Mode B — Stack design (flagship)
Walkthrough: goal "blue leaves when white-balanced on gravel." Try orange →
surface preview shows blue leaves but the green channel collapses → switch to
yellow → works. Narrow with advanced search, compare several yellows,
restrict to owned filters (no buying).

Requirements:
- **Transmission chart + WB'd RGB response chart are always visible** in the
  center; the Surfaces instrument shares the screen (tab strip below them).
  Today these live in separate scroll regions; that split is the core UX bug
  this rebuild fixes.
- Owned-only toggle sits in the stack picker *and* the shelf.
- Candidate comparison (sparkline rows, side-by-side) supports the bake-off.

### Mode C — Film matching (the missing mode)
Every component already exists — film profiles, emulation target, deviation
metrics, channel mixer — but nothing composes them, which is why the workflow
currently fails. Mode C is a dedicated instrument (§3, Film Match tab):

1. Pick a film target (its spectral RGB response curves).
2. Shape the achieved response with the filter stack (charts overlay achieved
   vs target, with deviation readouts).
3. Fine-tune relative channel intensities with the embedded channel mixer.
4. Export a report of the final match.

The channel mixer is demoted from a standalone panel into this instrument —
it was "rarely used" because it was homeless; here it has a job.

**Frequency ranking → placement rules:** filter search + combined plot + WB'd
RGB response = prime real estate, always visible. Owned-only = prominent.
Advanced search = first-class in the Filters shelf. Report = header toolbar.
Imports = Data shelf, rare but urgent when needed. My Filters manager = deep
collapsible (purchase-time only). Reflector search = redesigned (see §4) —
it is rare *because* it works poorly.

## 3. Information architecture

Three zones plus a header:

```
┌────────── Header: brand │ stack summary chips │ [Report] [Reset] ──────────┐
│ Setup rail (left)  │ Instrument canvas (center)    │ Shelves (right)       │
│  Sensor select     │ 1. Transmission chart+metrics │ Tabs:                 │
│  Film select       │ 2. Sensor response chart+WB   │  Filters              │
│  Illuminant select │ 3. [Surfaces|FilmMatch|Ref]   │  Reflectors           │
│  Stack editor      │                               │  Data                 │
└────────────────────┴───────────────────────────────┴───────────────────────┘
```

### 3.1 Setup rail (left drawer, open by default)
- **Sensor** select (camera QE profiles).
- **Film** select — `None` = digital sensor; selecting a film switches the
  response instrument to film dye-layer semantics (existing behavior).
- **Illuminant** select.
- **Stack editor:** ordered stack list with per-filter multiplier, remove,
  clear-all; inline quick-add search (name substring) with owned badge on
  results; **Owned only** switch restricting results to My Filters.

### 3.2 Instrument canvas (center)
1. **Transmission** — combined stack transmission (linear / log / stops
   toggle, chart-local) + metrics row (avg transmission %, effective stops).
2. **Sensor response** — per-channel response chart; WB toggle; R/G/B
   visibility toggles; WB gain readout; explicit indicator when WB is taken
   from a surface reference.
3. **Tab strip: [Surfaces | Film Match | Reference]** — charts 1–2 stay
   rendered; the tab strip swaps only the region below.
   - **Surfaces:** white-referenced color chip grid (§7.1). Per chip: swatch,
     name, coverage note, [WB] (use as WB reference), [×] (remove).
     "Add surface" routes to the Reflectors shelf.
   - **Film Match:** the Mode C instrument (§7.2).
   - **Reference:** raw QE curves, illuminant curve, film target curves —
     read-only lookup (demoted from the main flow).

### 3.3 Shelves (right drawer, tabbed: Filters | Reflectors | Data)
- **Filters:** quick search; advanced search (manufacturer multi-select,
  wavelength/transmission criteria rows, sort incl. Hex-Rainbow); owned-only;
  result rows with sparkline, owned badge, [add to stack]; multi-select
  compare; **My Filters manager** collapsed at the bottom (add/remove
  individual, per-manufacturer, clear all).
- **Reflectors:** library list ranked/filterable by *usefulness*
  (wavelength coverage / data-quality badge — useful spectra must be findable
  inside messy imports); click-to-preview chip before committing; actions
  [Add as surface] [Use as WB]; ECOSIS import with **preview-before-commit**;
  org/package/target search filters.
- **Data:** import cards for the existing importer set (filter CSV,
  digitized CSV, ECOSIS, illuminant, film profile, sensor QE), with the
  **Quick digitized filter** card first (Mode A path); each card shows
  last-import status.

## 4. Feature disposition

| # | Feature | Old home | New home | Disposition |
|---|---|---|---|---|
| 1 | Quick filter search | sidebar | stack editor + Filters shelf | keep (heart) |
| 2 | Advanced filter search | main content toggle | Filters shelf | keep |
| 3 | Owned-only toggle | sidebar checkbox | stack picker + shelf | keep, promoted |
| 4 | My Filters manager | main content panel | Filters shelf, collapsed | keep, demoted |
| 5 | Stack editor + multipliers | sidebar | Setup rail | keep |
| 6 | Transmission chart + metrics | main content | canvas §3.2.1 | keep (heart) |
| 7 | Stops/log view toggle | AppState `log_view` | chart-local (view layer) | keep, rehomed |
| 8 | WB toggle + gains display | main content | canvas §3.2.2 | keep |
| 9 | R/G/B channel visibility | AppState flags | chart-local (view layer) | keep, rehomed |
| 10 | Sensor response chart | main content | canvas §3.2.2 | keep (heart) |
| 11 | Surface color swatches | reflector list rows | Surfaces instrument, §7.1 math | keep, rebuilt |
| 12 | WB from surface | reflector row button | Surfaces chip [WB] | keep |
| 13 | Default reflector seeding | service | unchanged (service) | keep |
| 14 | Reflector advanced search | main content panel | Reflectors shelf | keep, redesigned (usefulness badges) |
| 15 | ECOSIS import | import dialog | Reflectors shelf | keep, redesigned (preview-before-commit) |
| 16 | Other importers (filter/digitized/illuminant/film/QE CSV) | import dialog | Data shelf | keep; digitized quick-add promoted |
| 17 | Film emulation target + film WB | sidebar section | Film Match tab | keep, promoted |
| 18 | Deviation metrics | ad-hoc view code | presenter `FilmMatchVM` (§7.2) | keep, rehomed |
| 19 | Channel mixer | standalone panel | embedded in Film Match | keep, demoted |
| 20 | Report export (PNG/TSV) | action | header toolbar | keep |
| 21 | Reset to defaults | sidebar button | header toolbar | keep |
| 22 | Raw QE / illuminant / film curves | main content | Reference tab | keep, demoted |
| 23 | Sidebar expansion persistence | AppState UI-state dict | dropped | replaced by shelf tabs; no persistence in v1 |

Nothing is cut. Items 7, 9, 19, 22 are relocations; 11, 14, 15 are redesigns;
17–19 compose into the new Film Match instrument.

## 5. Interaction & refresh contract

- Every control dispatches a named action:
  `views/actions.py :: handle_app_action(action, payload, ctx) -> set[str]`,
  where `ctx` bundles `(app_state, data)`. Actions mutate state, call
  services, and return the set of **dirty panel keys**.
- Panels register refresh callbacks in a view-layer `PanelRegistry` (plain
  dict: key → callable). The shell maps dirty keys → targeted refreshes;
  the key `"*"` means rebuild-everything.
- Panel keys: `setup`, `transmission`, `sensor_response`, `surfaces`,
  `film_match`, `reference`, `shelf_filters`, `shelf_reflectors`,
  `shelf_data`.
- **No full-page rebuild on ordinary interactions.** The current
  `_refresh()`-everything pattern (re-render sidebar + main content on every
  action) is retired; it is the root cause of focus loss, dropdown churn and
  scroll jumps. `"*"` is reserved for Reset and data-invalidating imports.
- **View-visibility state lives in the view layer**, in a `page_state` dict
  owned by the page builder: active tab, open/closed sections, dialog flags,
  and chart-local toggles (log/stops, R/G/B visibility). AppState keeps only
  what describes the user's *optical configuration* (selections, stack, WB
  reference, mixer settings, advanced-search parameters, My Filters) — never
  the user's scroll position. Legacy `AppState.show_*` fields become dead at
  U1 and are pruned in U6.

## 6. Seam rules (architecture contract)

- `views/` may import only: `nicegui`, `views.*`, `services.presenter`
  (VM dataclasses + `prepare_*` functions), `services.state_manager` (type
  only), `models.constants` (UI label constants), `views.actions`.
- `views/` must NOT import `services.calculations`, `services.visualization`,
  `services.spectra`, `services.data`, `services.importing`,
  `services.channel_mixer`, and must not compute with `numpy`/`pandas`/
  `plotly.graph_objects`. Figures arrive pre-built inside VMs; views render.
- `views/actions.py` may import any service. It contains no NiceGUI layout
  code — orchestration only (state mutation, service calls, dirty-set
  computation) plus user feedback via `ui.notify` (the one sanctioned UI call).
- `services/presenter.py` stays UI-toolkit-free (no `nicegui` import).
- `app.py` stays a thin shell: data init, AppState init, page scaffolding,
  registry wiring. No feature logic.

## 7. Computation changes (backend; each lands with tests)

### 7.1 White-referenced surface colors (U4)

**Problem (verified in code):** `views/main_content.py:117` calls
`prepare_rgb_for_display(raw_rgb, auto_exposure=True)`;
`services/visualization.py:149–153` scales each surface's brightest channel
to 1.0 independently → a near-black leaf `[0.02, 0.001, 0.001]` renders as
`[1.0, 0.05, 0.05]` (bright red), and chips are not mutually comparable.

**Fix: a shared white anchor.**
- Define `white_ref` = sensor response to a **perfect diffuse reflector**
  (reflectance = 1.0 at every wavelength) under the current
  stack × illuminant × sensor × WB (+ mixer). Physically: the swatches are
  "exposed so a perfect white just saturates."
- Display (linear): `chip_linear[c] = clip(response(surface)[c] / white_ref[c], 0, 1)`.
- Display (CSS): **apply the sRGB transfer function** (IEC 61966-2-1
  piecewise gamma) before hex conversion — browsers assume sRGB; without it
  midtones render too dark. This is the decided behavior. Both linear and
  encoded values are carried in the VM.
- Testable properties: black surface → black; the white anchor → exactly
  white; hue direction preserved; brightness monotonic in reflectance; the
  surface chosen as WB reference renders exactly white by construction.
- New code: `calculations.compute_surface_color(...)` (white-ref + clip) +
  sRGB encoding helper; `presenter.prepare_surfaces(app_state,
  reflector_collection) -> SurfacesVM` (chips: name, hex display, linear RGB,
  is_wb_ref, coverage/metadata).
- `visualization.prepare_rgb_for_display(auto_exposure=True)` remains for
  chart annotation use only; it must not be used for swatches.

### 7.2 Film-match consolidation (U5)

- Deviation-metric math currently embedded in view code moves into
  `presenter.prepare_film_match(...) -> FilmMatchVM`: achieved-vs-target
  overlay figure, per-channel peak wavelength, peak-response ratio, RMS
  deviation after peak normalization, film-WB gains.
- Unit tests cover the ported metric math (reference behavior recoverable
  from git history of the deleted `views/main_content.py`).
- Channel-mixer semantics unchanged (`services/channel_mixer.py` untouched);
  the mixer is embedded as a Mode C control.

Everything else in `services/` and `models/` is frozen.

## 8. Acceptance criteria (manual walkthrough gates)

**Mode A:** cold start → search "ND" → digitized-CSV import (≤2 clicks) →
add to stack → transmission + WB'd RGB response visible without scrolling
past setup → Report exports PNG/TSV. No dropdown churn or focus loss at any
step (the targeted-refresh proof).

**Mode B:** select sensor + illuminant → add surface "leaf" from the
Reflectors shelf → add an orange filter → leaf chip is **not** bright red
(white-referenced per §7.1) → toggle WB-from-surface → that chip renders
exactly white → restrict picker to owned → compare two yellow candidates via
sparklines → transmission + response + surfaces co-visible on one screen.

**Mode C:** pick film target → Film Match overlay shows achieved vs target
curves → deviation readout updates on every stack change → mixer sliders
re-shape achieved curves → report reflects the match.

**Global:** the unittest suite stays green after every slice; ordinary
actions refresh only the panels in their dirty set (never `*`); switching
canvas tabs must not re-run service computations for charts 1–2
(re-render at most).

## 9. Rebuild slices & gates

| Slice | Ships | Gate |
|---|---|---|
| A1 | this spec | reviewed |
| U1 | delete `views/*.py` + `actions.py`; stub `app.py` (boots to placeholder) | 95/95 green; `python program/app.py` boots |
| U2 | shell + PanelRegistry + Setup rail (sensor/film/illuminant/stack editor) + Transmission + metrics + Filters shelf (quick search, owned, add, My Filters collapsed) | boot + Mode A partial |
| U3 | Sensor response chart + WB + channel toggles + Reference tab + Report toolbar action | boot + WB flows |
| U4 | Surfaces instrument + §7.1 math + tests + Reflectors shelf (library, redesigned search, ECOSIS preview import) | §7.1 tests green + Mode B walkthrough |
| U5 | Film Match instrument + §7.2 `FilmMatchVM` + tests + embedded mixer | §7.2 tests green + Mode C walkthrough |
| U6 | theming; prune dead `constants.py` strings + dead `AppState.show_*` fields; USAGE/README rewrite | suite green; docs match UI |

S7 (pandas removal from presenter search) remains deferred until after U6.

## 10. Theming & visual language

- Quasar dark palette as default; Inter/system-ui font retained.
- Swatches: ≥36 px, 1 px neutral border (near-black chips must read against
  the card background), hex + name label.
- Charts: keep existing per-channel color constants (R/G/B semantic colors);
  target overlays dashed.
- Feedback via `ui.notify` (success/error), never raw tracebacks.

## 11. Docs

- U6 rewrites `USAGE.md` from the new UI (task-oriented; the three modes as
  chapters).
- `README.md` points to `UX_SPEC.md` for design rationale; `TECHNICAL.md`
  gains the §5–§7 contracts.

*Adopted: 2026-09-13, `ui-rebuild` branch.*



