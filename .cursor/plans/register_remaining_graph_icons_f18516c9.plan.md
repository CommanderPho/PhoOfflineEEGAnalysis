---
name: Register remaining graph icons
overview: Extend [`pypho_timeline/EXTERNAL/pyqtgraph/icons/__init__.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\icons\__init__.py) with `GraphIcon(...)` module-level entries for every image file in that folder that is not already listed, using valid Python identifiers for hyphenated names. Resolve the `icons.png` / `icons.svg` registry key collision before or during the edit.
todos:
  - id: resolve-icons-collision
    content: Choose rename vs single registration vs optional registry_name for icons.png / icons.svg
    status: completed
  - id: append-graphicon-lines
    content: Add GraphIcon lines for all unregistered assets in icons/__init__.py (camelCase vars, one line each)
    status: completed
  - id: verify-import
    content: Smoke-test import and getGraphIcon for a few keys including a hyphenated name
    status: completed
isProject: false
---

# Register remaining icons in `icons/__init__.py`

## Context

- [`GraphIcon`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\icons\__init__.py) registers each asset under the **filename stem** (text before the first `.`), e.g. `GraphIcon("ui-toolbar.png")` enables `getGraphIcon("ui-toolbar")`.
- The file currently ends with only five registrations (`auto`, `ctrl`, `default`, `invisibleEye`, `lock`), matching [upstream pyqtgraph](https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/icons/__init__.py).
- Your local folder (verified via directory listing) contains **20 additional** raster/SVG assets that need `GraphIcon` lines.

## Files already registered (do not duplicate)

`auto.png`, `ctrl.png`, `default.png`, `invisibleEye.svg`, `lock.png`

## Assets to add (one `GraphIcon("filename")` per line)

Use **camelCase** module variables to match existing `invisibleEye`, and map hyphenated filenames to readable names:

| File | Suggested variable | `getGraphIcon` key |
|------|-------------------|-------------------|
| `application-dock-tab.png` | `applicationDockTab` | `application-dock-tab` |
| `application-rename.png` | `applicationRename` | `application-rename` |
| `application-tile-horizontal.png` | `applicationTileHorizontal` | `application-tile-horizontal` |
| `application-wave.png` | `applicationWave` | `application-wave` |
| `category-group.png` | `categoryGroup` | `category-group` |
| `commentts.png` | `commentts` | `commentts` |
| `film-timeline.png` | `filmTimeline` | `film-timeline` |
| `film.png` | `film` | `film` |
| `films.png` | `films` | `films` |
| `form.png` | `form` | `form` |
| `globe-green.png` | `globeGreen` | `globe-green` |
| `log_scroll.png` | `log_scroll` | `log_scroll` |
| `settings_gear.png` | `settings_gear` | `settings_gear` |
| `settings_wrench.png` | `settings_wrench` | `settings_wrench` |
| `settings.png` | `settings` | `settings` |
| `sticky-notes-stack.png` | `stickyNotesStack` | `sticky-notes-stack` |
| `table-heatmap.png` | `tableHeatmap` | `table-heatmap` |
| `table.png` | `table` | `table` |
| `tables-stacks.png` | `tablesStacks` | `tables-stacks` |
| `ui-toolbar.png` | `uiToolbar` | `ui-toolbar` |

Alphabetical ordering by filename (or by variable name) keeps the block easy to diff later.

## Collision: `icons.png` vs `icons.svg`

Both stems resolve to registry key **`icons`**; the second `GraphIcon` would overwrite the first.

Pick **one** approach (minimal scope first):

1. **Rename on disk (recommended if both files must be loadable):** Rename e.g. `icons.png` to `icons-raster.png` (or another unique stem), then add `iconsRaster = GraphIcon("icons-raster.png")` and `icons = GraphIcon("icons.svg")` (or keep SVG as `icons` and give the PNG a distinct stem).
2. **Register only one:** If only one asset is needed in the app, add a single `GraphIcon` for that file and omit the other (or delete the unused file separately).
3. **Small API extension (only if you refuse renames):** Add an optional `registry_name` (or similar) to `GraphIcon.__init__` so two files can map to different keys. This is more invasive than (1) and conflicts with “minimal edits” unless you already need custom keys elsewhere.

No code in the repo currently calls `getGraphIcon('icons')` (only `'default'` is used from this registry in vendored pyqtgraph), so the choice is mainly about future use and keeping both files addressable.

## What not to change

- **`__all__`**: Still only `getGraphIcon` / `getGraphPixmap` unless you explicitly want icon constants exported.
- **Behavior of** `GraphIcon` / `getGraphIcon`: Unchanged unless you choose collision option (3).

## Verification

After edits, run a quick import check (e.g. `python -c "from pypho_timeline.EXTERNAL.pyqtgraph.icons import getGraphIcon; getGraphIcon('ui-toolbar')"` with your package path) or open the icon preview widget if you use one, to confirm keys match expectations.
