---
name: Show table QIcon from pg icons
overview: "`DockButtonConfig` only supports `QStyle.StandardPixmap` today; the Dock applies `standardIcon(cfg.buttonIcon)`. To use the bundled `table.png` registration (`getGraphIcon('table')`), add an optional `QIcon` override on `DockButtonConfig` and branch in `_buildCustomButtons`, then wire `timeline_builder` to pass that icon for the `show_table` button."
todos:
  - id: dock-button-qicon
    content: Add buttonQIcon to DockButtonConfig and branch in _buildCustomButtons (Dock.py)
    status: completed
  - id: timeline-builder-table-icon
    content: Import getGraphIcon and use buttonQIcon=getGraphIcon('table') in both branches (timeline_builder.py)
    status: completed
isProject: false
---

# Use pyqtgraph `table` icon for show_table dock button

## Why two files

[`DockButtonConfig`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\dockarea\Dock.py) types `buttonIcon` as `QtWidgets.QStyle.StandardPixmap`. [`_buildCustomButtons`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\dockarea\Dock.py) (lines ~1200–1212) always calls `QtWidgets.QApplication.style().standardIcon(cfg.buttonIcon)`. You cannot pass [`getGraphIcon('table')`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\icons\__init__.py) without extending this API.

## 1. Extend `DockButtonConfig` in [Dock.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\dockarea\Dock.py)

After the existing `buttonIcon` field, add:

- `buttonQIcon: Optional[QtGui.QIcon] = field(default=None, ...)` (short metadata desc optional)

`Optional` is already imported from `typing`; `QtGui` is already imported from `..Qt`.

## 2. Update `_buildCustomButtons` in the same file

For both `setIcon` call sites in the `for key, cfg in custom_button_configs.items():` loop, use:

- `_icon = cfg.buttonQIcon if getattr(cfg, 'buttonQIcon', None) is not None else QtWidgets.QApplication.style().standardIcon(cfg.buttonIcon)`
- `btn.setIcon(_icon)`

(Or inline equivalent; avoid duplicating logic if you extract a small local helper.)

## 3. Update [timeline_builder.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\timeline_builder.py)

- Import: `from pypho_timeline.EXTERNAL.pyqtgraph.icons import getGraphIcon` (next to the existing `DockButtonConfig` import is fine).

- Replace both `DockButtonConfig(..., buttonIcon=QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView, ...)` branches (lines ~1392 and ~1395) with:

`DockButtonConfig(showButton=True, buttonQIcon=getGraphIcon('table'), buttonToolTip='Show table')`

This matches the registry key for `table = GraphIcon("table.png")` in [`icons/__init__.py`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\icons\__init__.py).

## 4. Optional cleanup

[`buttonIcon_ShowTable`](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\EXTERNAL\pyqtgraph\dockarea\Dock.py) (line ~207) is a separate constant dict; no change required unless something else reads it for `show_table`.

## Verification

- Grep confirms only these two `DockButtonConfig` sites in `timeline_builder.py`.
- Run a quick UI smoke test: open a dock with `detailed_df` and confirm the title-bar table button shows the bundled PNG.
