---
name: Remove irrelevant controls from TopoMNEControlPanel
overview: Hide the Lower Limit, Upper Limit, and Highpass Cutoff controls in TopoMNEControlPanel since they are not relevant for the TopoMNE renderer.
todos: []
---

# Remove Irrelevant Controls from TopoMNEControlPanel

## Overview

The `TopoMNEControlPanel` inherits from `IControlPanel`, which automatically creates widgets for "Lower Limit", "Upper Limit", and "Highpass Cutoff". These controls are not relevant for the TopoMNE renderer and should be hidden.

## Implementation

### File to Modify

- [`stream_viewer/stream_viewer/widgets/topo_mne_ctrl.py`](stream_viewer/stream_viewer/widgets/topo_mne_ctrl.py)

### Changes

1. **In `_init_widgets()` method** (after line 17, after calling `super()._init_widgets()`):

- Hide the "Lower Limit" widget and its label by finding the spinbox with object name `"LL_SpinBox"` and its corresponding label, then calling `setVisible(False)` on both
- Hide the "Upper Limit" widget and its label by finding the spinbox with object name `"UL_SpinBox"` and its corresponding label, then calling `setVisible(False)` on both
- Hide the "Highpass Cutoff" widget and its label by finding the spinbox with object name `"HP_SpinBox"` and its corresponding label, then calling `setVisible(False)` on both

2. **In `reset_widgets()` method** (after line 124, after calling `super().reset_widgets(renderer)`):

- Skip the connection logic for these three widgets (the base class `reset_widgets` will still try to connect them, but since they're hidden, this is acceptable. Alternatively, we could override to prevent the connection, but hiding is sufficient).

### Implementation Details

- Use `self.findChild()` to locate each spinbox by its object name
- Find the corresponding label by getting the widget at the same row position (column 0) in the grid layout
- Call `setVisible(False)` on both the label and the spinbox
- The widgets will remain in the layout but won't be visible to the user