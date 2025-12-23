---
name: Update TopoMNE to PyVista 3D Head Mesh
overview: Replace the matplotlib-based TopoMNE renderer with a PyVista-based 3D head mesh visualization that matches the working notebook implementation, including real-time data updates and control widgets for configuration.
todos:
  - id: create_pyvista_renderer
    content: Create PyVistaRenderer base class in display/pyvista.py with Qt integration and timer support
    status: completed
  - id: update_topo_mne_core
    content: Update TopoMNE to inherit from PyVistaRenderer and implement mesh loading, electrode positioning, and visualization setup
    status: completed
    dependencies:
      - create_pyvista_renderer
  - id: implement_realtime_updates
    content: Implement update_visualization() method to color electrode glyphs based on incoming data values
    status: completed
    dependencies:
      - update_topo_mne_core
  - id: add_control_properties
    content: Add properties for head_mesh_path, mesh_opacity, cone dimensions, and electrode offsets with setters that trigger reset_renderer
    status: completed
    dependencies:
      - update_topo_mne_core
  - id: create_control_panel
    content: Create TopoMNEControlPanel widget with file pickers, spinboxes, and checkboxes for all configuration options
    status: completed
    dependencies:
      - add_control_properties
  - id: register_control_panel
    content: Add COMPAT_ICONTROL class variable to TopoMNE and ensure control panel is discoverable
    status: completed
    dependencies:
      - create_control_panel
---

# Update TopoMNE Renderer to Use PyVista 3D Head Mesh

## Overview

Transform the `TopoMNE` renderer from a 2D matplotlib-based visualization to a 3D PyVista-based head mesh renderer that matches the working notebook implementation. The renderer will display a 3D head mesh with electrode positions, support real-time data updates, and include control widgets for configuration.

## Architecture Changes

### 1. Create PyVista Display Renderer Base Class

- **File**: `stream_viewer/stream_viewer/renderers/display/pyvista.py` (new)
- Create `PyVistaRenderer` class similar to `MPLRenderer` but using `pvqt.BackgroundPlotter()`
- Implement container widget management for Qt integration
- Handle timer-based updates for real-time visualization

### 2. Update TopoMNE Renderer

- **File**: `stream_viewer/stream_viewer/renderers/topo_mne.py`
- Change inheritance from `MPLRenderer` to `PyVistaRenderer`
- Implement notebook logic:
- Load STL head mesh from path (with default fallback)
- Scale mesh from mm to meters if needed
- Center mesh at origin
- Load electrode montage (existing logic)
- Compute normals and find nearest surface points for each electrode
- Create PyVista visualization with:
    - Head mesh (semi-transparent)
    - Cone glyphs at electrode positions (oriented along normals)
    - Text labels for electrode names
- Add real-time data update support:
- Color electrode glyphs based on incoming data values
- Optionally scale glyph size based on data
- Add properties for:
- `head_mesh_path` (with default)
- `mesh_opacity`
- `cone_radius` and `cone_height`
- `show_labels`
- `electrode_offset_y` and `electrode_offset_z` (for manual adjustment)

### 3. Create Control Panel Widget

- **File**: `stream_viewer/stream_viewer/widgets/topo_mne_ctrl.py` (new)
- Extend `IControlPanel` or `MPLRenderer` control panel
- Add widgets for:
- Head mesh path (QLineEdit + QPushButton for file picker)
- Montage path (QLineEdit + QPushButton, reuse existing montage_path logic)
- Mesh opacity (QDoubleSpinBox)
- Cone size controls (radius, height)
- Show labels checkbox
- Electrode offset controls (Y, Z adjustments)

### 4. Update Renderer Registration

- **File**: `stream_viewer/stream_viewer/renderers/topo_mne.py`
- Add `COMPAT_ICONTROL = ['TopoMNEControlPanel']` class variable
- Ensure control panel is discoverable by the widget system

## Implementation Details

### Default Head Mesh Path

Use the path from the notebook as default:

```python
default_head_mesh_path = Path(r"C:/Users/pho/repos/EmotivEpoc/PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/resources/ElectrodeLayouts/head_bem_1922V_fill.stl")
```



### Real-time Data Updates

In `update_visualization()`:

- Map data values to colors using colormap
- Update electrode glyph colors based on current data
- Optionally update glyph sizes if enabled

### Error Handling

- Gracefully handle missing mesh files (show placeholder message)
- Handle missing montage (fallback to default)
- Handle mesh loading errors
- Handle PyVista initialization failures

## Files to Modify/Create

1. **Create**: `stream_viewer/stream_viewer/renderers/display/pyvista.py`
2. **Modify**: `stream_viewer/stream_viewer/renderers/topo_mne.py`
3. **Create**: `stream_viewer/stream_viewer/widgets/topo_mne_ctrl.py`

## Dependencies

- Ensure `pyvista` and `pyvistaqt` are available (already in dependencies based on grep results)
- Reuse existing `ElectrodeHelper` from `phoofflineeeganalysis`

## Testing Considerations

- Test with default paths
- Test with custom mesh and montage paths