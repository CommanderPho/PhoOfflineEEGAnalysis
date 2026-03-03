---
name: Text track top alignment
overview: Change text labels on timeline tracks from vertically centered to top-aligned by updating CustomRectBoundedTextItem's updatePosition() in the embed AlignableTextItem module.
todos: []
isProject: false
---

# Text-based track labels: center to top alignment

## Root cause

Text items on interval/track rectangles are positioned in [pypho_timeline/_embed/AlignableTextItem.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/_embed/AlignableTextItem.py) by `CustomRectBoundedTextItem.updatePosition()` (lines 361–378). It currently:

- Uses the rect **center** for position: `a_center_point = a_rect.center()` and `setPos(a_center_point.x(), a_center_point.y())`
- Sets text **anchor** to center: `setAnchor(pg.Point(0.5, 0.5))`

So the text is drawn with its center at the rect center, which matches the “centered in track” behavior in your screenshot.

## Data flow (where labels come from)

- **IntervalRectsItem** ([interval_rects_item.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/rendering/graphics/interval_rects_item.py)) builds label text via `format_label_fn` (or `rect_data_tuple.label`) and creates one **CustomRectBoundedTextItem** per interval in `rebuild_label_items()` (lines 266–310), passing the interval rect and calling `a_text_item.updatePosition()`.
- **CustomRectBoundedTextItem** is the only place that decides vertical alignment; callers do not pass an alignment option.

## Change to make

**File:** [pypho_timeline/_embed/AlignableTextItem.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/_embed/AlignableTextItem.py)

In `CustomRectBoundedTextItem.updatePosition()`:

1. **Anchor:** use top-center so the text’s top is fixed at the position:
  - `setAnchor(pg.Point(0.5, 0))`  
   (pyqtgraph: 0 = top, 1 = bottom; (0.5, 0) = top-center.)
2. **Position:** use the top-center of the rect instead of the center:
  - Replace `a_center_point = a_rect.center()` and `setPos(a_center_point.x(), a_center_point.y())` with the top of the rect, e.g. `pos_x = a_rect.center().x()`, `pos_y = a_rect.top()` (or equivalent for the top edge in data coords), then `setPos(pos_x, pos_y)`.

Result: the top-center of each label sits at the top-center of its interval rect, so text-based tracks align to the top of the track.

## Optional: make alignment configurable

If you want to keep center as an option (e.g. for other track types), you could:

- Add an optional constructor argument to `CustomRectBoundedTextItem`, e.g. `vertical_anchor='top'` or `vertical_anchor='center'` (default `'top'`).
- In `updatePosition()`, set anchor and position from that flag (e.g. `(0.5, 0)` and `a_rect.top()` for top, current behavior for center).

Then in [interval_rects_item.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/rendering/graphics/interval_rects_item.py), when creating `CustomRectBoundedTextItem` in `rebuild_label_items()`, pass the chosen alignment (or leave default as top). This adds a small amount of code but keeps behavior consistent and leaves room for future track-specific alignment without further refactors.

## Summary


| Location                                                                     | Change                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_embed/AlignableTextItem.py` → `CustomRectBoundedTextItem.updatePosition()` | Use anchor `(0.5, 0)` and position at rect top-center so labels align to top of track. Optionally add a `vertical_anchor` (or similar) parameter and branch on it if you want to keep center as an option. |


No changes are required in `IntervalRectsItem`, `TrackRenderer`, or `Render2DEventRectanglesHelper` unless you add the optional constructor parameter and pass it from `rebuild_label_items()`.