---
name: Timeline Core Concepts Refactor
overview: Extract core timeline rendering concepts and standardize overview/detailed mode logic across all tracks. Overview mode shows only interval rectangles; detailed mode shows interval rectangles plus detailed overlays.
todos:
  - id: base-track-standardization
    content: "Update BaseTrackWidget: standardize _render_overview() and _render_detailed() to ensure interval rectangles are always rendered, add helper method for clarity"
    status: completed
  - id: motion-track-fix
    content: "Fix MotionRecordingTrack: show interval rectangles + line plots in detailed mode, ensure rectangles are visible"
    status: completed
    dependencies:
      - base-track-standardization
  - id: pholog-track-fix
    content: "Fix PhoLogTrack: move text labels to detailed mode only, ensure interval rectangles are always shown"
    status: completed
    dependencies:
      - base-track-standardization
  - id: verify-other-tracks
    content: Verify and fix other tracks (EEG, Video, String, XDF, Whisper) to follow consistent pattern
    status: completed
    dependencies:
      - base-track-standardization
  - id: test-rendering
    content: Test overview/detailed mode transitions and verify all tracks render correctly
    status: completed
    dependencies:
      - motion-track-fix
      - pholog-track-fix
      - verify-other-tracks
---

# Timeline Core Concepts Refactor

## Overview

Refactor the timeline system to extract core rendering concepts and ensure consistent overview/detailed mode behavior across all tracks. The key principle is:

- **Overview mode**: Render interval rectangles only (no detailed overlays)
- **Detailed mode**: Render interval rectangles + detailed overlay (both visible simultaneously)

## Core Concepts

### 1. Interval Rectangle Rendering

All tracks render interval rectangles using `bar_graph_item` (pg.BarGraphItem) in the base `_render_overview()` method. This is the core visualization that shows when each epoch/interval exists.

### 2. Detailed Overlay Rendering

Tracks that support detailed mode add additional visualizations (line plots, text labels, etc.) on top of the interval rectangles when zoomed in.

### 3. Mode Switching Logic

The mode is determined by comparing the visible time span to `detailed_mode_timespan_threshold_sec`:

- If visible span ≤ threshold: detailed mode
- If visible span > threshold: overview mode

## Current Issues

1. **MotionRecordingTrack**: Hides interval rectangles in detailed mode (line 165: `self.bar_graph_item.setVisible(False)`)
2. **PhoLogTrack**: Renders text labels in both overview and detailed modes (should only be in detailed mode)
3. **Inconsistent behavior**: Some tracks don't properly separate overview vs detailed rendering

## Implementation Plan

### Phase 1: Base Track Widget Standardization

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/BaseTrackWidget.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/BaseTrackWidget.py)

1. **Update `_render_overview()` documentation**:

- Clarify that this method renders ONLY interval rectangles
- Should not include any detailed overlays

2. **Update `_render_detailed()` default implementation**:

- Change from calling `_render_overview()` to:
    - First call `super()._render_overview(time_range)` to render interval rectangles
    - Then call `_clear_detailed_items()` to ensure clean state
    - Then subclasses can add their detailed overlays
- This ensures interval rectangles are always visible in detailed mode

3. **Add helper method `_render_interval_rectangles()`**:

- Extract the core rectangle rendering logic from `_render_overview()`
- This can be called by both overview and detailed modes
- Makes the intent explicit

### Phase 2: MotionRecordingTrack Fix

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/MotionRecordingTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/MotionRecordingTrack.py)

1. **Update `_render_detailed()`**:

- Remove `self.bar_graph_item.setVisible(False)` (line 165)
- Call `super()._render_overview(time_range)` first to render interval rectangles
- Then render line plots as overlay
- Ensure both are visible simultaneously

2. **Update `_render_overview()`**:

- Ensure it calls `_clear_detailed_items()` to hide line plots
- Call `super()._render_overview(time_range)` for rectangles

### Phase 3: PhoLogTrack Fix

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

1. **Update `_render_overview()`**:

- Remove call to `_render_text_labels_and_points()` (line 408)
- Only render interval rectangles (call `super()._render_overview(time_range)`)
- Call `_clear_detailed_items()` to hide text labels

2. **Update `_render_detailed()`**:

- Call `super()._render_overview(time_range)` to render interval rectangles
- Then call `_render_text_labels_and_points(time_range)` to add text overlay
- Both should be visible

3. **Implement `_clear_detailed_items()`**:

- Clear text items and point markers
- Remove from plot widget

4. **Implement `_ensure_detailed_items()`**:

- Ensure text rendering infrastructure is ready (may already be in place)

### Phase 4: Other Tracks Verification

**Files to check**:

- [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/EEGRecordingTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/EEGRecordingTrack.py)
- [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/VideoMetadataTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/VideoMetadataTrack.py)
- [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/StringDataTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/StringDataTrack.py)
- [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/XDFStreamTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/XDFStreamTrack.py)
- [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/WhisperTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/WhisperTrack.py)

**Actions**:

1. Verify all tracks properly implement `_clear_detailed_items()` (even if empty)
2. Verify all tracks properly implement `_ensure_detailed_items()` (even if empty)
3. Ensure no tracks override `_render_overview()` to include detailed overlays
4. Ensure tracks that don't support detailed mode have `detailed_mode_timespan_threshold_sec = None`

## Implementation Details

### Core Rendering Pattern

```python
def _render_overview(self, time_range):
    """Render ONLY interval rectangles - no detailed overlays."""
    # Clear any detailed items
    self._clear_detailed_items()
    # Render interval rectangles
    super()._render_overview(time_range)  # or _render_interval_rectangles()

def _render_detailed(self, time_range):
    """Render interval rectangles + detailed overlay."""
    # First render interval rectangles
    super()._render_overview(time_range)  # or _render_interval_rectangles()
    # Then add detailed overlay
    self._ensure_detailed_items()
    # ... render detailed items ...
```



### Mode Detection Logic

The existing logic in `update_display()` is correct:

- Compares visible time span to `detailed_mode_timespan_threshold_sec`
- Sets `_is_detailed_mode` flag
- Calls appropriate render method