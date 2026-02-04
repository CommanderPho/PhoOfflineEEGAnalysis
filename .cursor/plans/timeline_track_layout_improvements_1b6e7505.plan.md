---
name: Timeline track layout improvements
overview: Modify BaseTrackWidget to make plots fill vertical space, rotate track labels vertically, and reduce label width for a more compact layout.
todos:
  - id: remove-fixed-height
    content: Change plot_widget from setFixedHeight to setMinimumHeight to allow vertical expansion
    status: completed
  - id: rotate-label
    content: Implement -90 degree rotation for name_label text using QTransform or custom paintEvent
    status: completed
  - id: reduce-label-width
    content: Reduce name_label fixed width from 80 to 30-35 pixels for more compact left bar
    status: completed
    dependencies:
      - rotate-label
---

# Timeline Track Layout Improvements

## Overview

Modify the `BaseTrackWidget` class to improve the layout of timeline tracks:

1. Make plots fill the full vertical space of their container
2. Rotate track name labels -90 degrees (vertical text)
3. Reduce the label width to make the left bar smaller

## Changes Required

### File: `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/BaseTrackWidget.py`

**Change 1: Remove fixed height constraint on plot widget**

- Currently: `self.plot_widget.setFixedHeight(height)` on line 62
- Change to: Remove the fixed height constraint and let the plot widget expand vertically
- The plot widget should use `setMinimumHeight(height)` instead to maintain a minimum size while allowing expansion

**Change 2: Rotate label text -90 degrees**

- Currently: `self.name_label = QLabel(name, self)` with normal horizontal text on line 104
- Change to: Rotate the label text -90 degrees using QTransform or QLabel's transformation
- Use `QTransform` to rotate the label's painter, or use a QLabel with rotated text rendering

**Change 3: Reduce label width**

- Currently: `self.name_label.setFixedWidth(80)` on line 106
- Change to: Reduce to a smaller width (e.g., 30-40 pixels) to make the left bar more compact
- Adjust font size if needed to fit rotated text

## Implementation Details

1. **Plot Height**: Change from `setFixedHeight(height)` to `setMinimumHeight(height)` to allow vertical expansion while maintaining a minimum size.