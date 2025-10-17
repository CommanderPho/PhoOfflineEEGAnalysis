## ADDED Requirements

### Requirement: Standalone Results Browser Export
The system SHALL export a single, self-contained `.html` artifact that can be opened locally in a modern browser without any server.

#### Scenario: Generate artifact from sessions
- **WHEN** the user runs the export API/CLI with one or more sessions
- **THEN** the system SHALL produce a single `.html` file containing UI and embedded data references

#### Scenario: Offline viewing
- **WHEN** the artifact is opened without internet connectivity
- **THEN** all UI, scripts, and styles SHALL function without external CDNs

#### Scenario: Data embedding and references
- **WHEN** datasets exceed configurable thresholds
- **THEN** the artifact SHALL inline small data and reference larger assets (e.g., `.nc` or `.zarr`) via relative paths with a clear missing-data message if unavailable

#### Scenario: Multi-session navigation
- **WHEN** multiple sessions are exported
- **THEN** the artifact SHALL allow selecting a session and modality, and update linked views accordingly

#### Scenario: Linked views
- **WHEN** a user selects a channel, time window, or annotation
- **THEN** spectrogram, bandpower, and events views SHALL synchronize to the selection

#### Scenario: Accessibility and performance
- **WHEN** rendering large spectrograms
- **THEN** the artifact SHALL use level-of-detail (LOD) and downsampling strategies to keep interactions responsive (<150ms median on typical laptop)

### Requirement: Export API and CLI
The system SHALL provide both a Python API and a CLI to generate the artifact.

#### Scenario: Python API
- **WHEN** `export_results_browser(sessions=[...], output_path=...)` is called
- **THEN** the export SHALL complete successfully and return the output path

#### Scenario: CLI usage
- **WHEN** the user runs `pho-eeg export-results-browser --input <path|glob|zarr> --out results.html`
- **THEN** the command SHALL create the artifact and exit with code 0 on success

### Requirement: Validation and schema
The export process SHOULD validate presence of required metadata (fs, channel names, time bounds) and warn when optional assets are missing.

#### Scenario: Missing optional data
- **WHEN** transcripts are not available
- **THEN** the artifact SHALL still render and omit the transcripts panel with a notice


### Requirement: Four-Panel Layout
The artifact SHALL render a four-panel UI: left sidebar (sessions), right sidebar (display options), bottom panel (comments), and main panel (timeline). All panels SHALL be visible concurrently on typical desktop resolutions.

#### Scenario: Panels render and initialize
- **WHEN** the artifact loads
- **THEN** the left sessions sidebar, right options sidebar, bottom comments panel, and main timeline panel SHALL be visible and initialized

#### Scenario: Session paging updates views
- **WHEN** the user changes the active session via the left sidebar
- **THEN** the main timeline, comment track, and applicable options SHALL update to the selected session

#### Scenario: Options update visuals
- **WHEN** the user toggles display options (channels, plots, overlays)
- **THEN** the main panel SHALL update without full-page reload

#### Scenario: Comments panel displays file content
- **WHEN** a comments file is provided
- **THEN** the bottom panel SHALL display the comments with timestamps and text

#### Scenario: Comment selection seeks timeline
- **WHEN** the user selects a comment in the bottom panel
- **THEN** the main timeline SHALL seek/center to that timestamp and highlight the corresponding marker in the comment track

#### Scenario: Timeline selection focuses comment
- **WHEN** the user clicks a comment marker in the main timeline
- **THEN** the bottom comments panel SHALL scroll to and highlight the corresponding entry

#### Scenario: Time-synchronized interactions
- **WHEN** the user zooms, pans, or brushes a time range in the main panel
- **THEN** raw EEG, result plots, and the comment track SHALL remain synchronized and reflect the same visible time window

#### Scenario: Main panel contents
- **WHEN** a session is active
- **THEN** the main panel SHALL show raw EEG traces for selected channels, the chosen result plots (e.g., spectrograms, bandpower trends), and a comment track aligned on the same time axis


