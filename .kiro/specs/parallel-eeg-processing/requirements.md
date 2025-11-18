# Requirements Document

## Introduction

This specification defines the refactoring of EEG processing and visualization workflows into reusable, parallelizable structures. Currently, the processing of .xdf files and generation of spectrograms/visualizations is tightly coupled within notebook code. This refactoring will enable batch processing of multiple recordings in parallel, improve code reusability, and support both programmatic and CLI-based workflows.

## Glossary

- **XDF File**: Lab Recorder eXtensible Data Format file containing multi-stream EEG recordings
- **Processing Pipeline**: The sequence of operations that transform raw EEG data into analysis outputs
- **Spectrogram**: Time-frequency representation of EEG signal power
- **Session**: A single EEG recording session, typically stored as one or more .xdf or .fif files
- **SavedSessionsProcessor**: Existing orchestrator class for multi-modality discovery and preprocessing
- **Worker**: A parallelizable unit of computation that processes a single session independently

## Requirements

### Requirement 1: Modular Processing Pipeline

**User Story:** As a researcher, I want to process EEG recordings through a well-defined pipeline, so that I can reuse processing steps across different analysis workflows

#### Acceptance Criteria

1. WHEN the system processes an XDF file, THE Processing Pipeline SHALL execute discrete stages (load, preprocess, analyze, visualize, export)
2. THE Processing Pipeline SHALL accept configuration parameters for each stage without requiring code modification
3. WHEN a processing stage fails, THE Processing Pipeline SHALL log the error and continue with remaining sessions
4. THE Processing Pipeline SHALL return structured results containing file paths, metadata, and processing status for each session
5. WHERE custom processing steps are needed, THE Processing Pipeline SHALL support injection of user-defined functions at each stage

### Requirement 2: Parallel Session Processing

**User Story:** As a researcher with multiple recording sessions, I want to process them in parallel, so that I can reduce total processing time

#### Acceptance Criteria

1. WHEN multiple XDF files are provided, THE System SHALL process them concurrently using available CPU cores
2. THE System SHALL limit concurrent workers to prevent memory exhaustion
3. WHILE processing sessions in parallel, THE System SHALL maintain independent state for each session
4. THE System SHALL aggregate results from all parallel workers into a unified output structure
5. IF a worker fails, THEN THE System SHALL continue processing remaining sessions and report the failure

### Requirement 3: Reusable Processing Functions

**User Story:** As a developer, I want processing logic separated from notebook code, so that I can use the same functions in scripts, notebooks, and CLI tools

#### Acceptance Criteria

1. THE System SHALL provide pure functions for each processing step that accept inputs and return outputs without side effects
2. WHEN processing functions are called, THE System SHALL not depend on global state or notebook-specific variables
3. THE System SHALL expose processing functions through a public API module
4. THE System SHALL provide type hints for all public function parameters and return values
5. WHERE file I/O is required, THE System SHALL accept Path objects and return Path objects for output files

### Requirement 4: Batch Spectrogram Generation

**User Story:** As a researcher, I want to generate spectrograms for all sessions in a directory, so that I can quickly review multiple recordings

#### Acceptance Criteria

1. WHEN a directory path is provided, THE System SHALL discover all XDF files recursively
2. THE System SHALL generate spectrograms for each discovered session
3. THE System SHALL support multiple output formats (HTML, PDF, PNG)
4. THE System SHALL apply consistent visualization parameters across all sessions
5. WHERE output files already exist, THE System SHALL skip regeneration unless explicitly requested

### Requirement 5: CLI Interface for Batch Processing

**User Story:** As a researcher, I want to process recordings from the command line, so that I can automate workflows without writing Python code

#### Acceptance Criteria

1. THE System SHALL provide a CLI command that accepts input directory and output directory paths
2. WHEN the CLI command is invoked, THE System SHALL process all XDF files in the input directory
3. THE System SHALL support command-line flags for parallelism level, output format, and processing options
4. THE System SHALL display progress information during batch processing
5. WHEN processing completes, THE System SHALL exit with code 0 on success and non-zero on failure

### Requirement 6: Configuration Management

**User Story:** As a researcher, I want to save processing configurations, so that I can reproduce analyses with consistent parameters

#### Acceptance Criteria

1. THE System SHALL accept configuration from Python dictionaries, JSON files, or YAML files
2. THE System SHALL validate configuration parameters before processing begins
3. WHEN invalid configuration is provided, THE System SHALL raise descriptive errors indicating the problem
4. THE System SHALL support configuration profiles for common analysis scenarios
5. THE System SHALL log the configuration used for each processing run

### Requirement 7: Result Aggregation and Export

**User Story:** As a researcher, I want processing results aggregated across sessions, so that I can perform cross-session analyses

#### Acceptance Criteria

1. WHEN batch processing completes, THE System SHALL generate a summary report with processing statistics
2. THE System SHALL export aggregated metadata as a pandas DataFrame
3. THE System SHALL support export of aggregated results to CSV, HDF5, and Zarr formats
4. THE System SHALL preserve session-level results alongside aggregated outputs
5. WHERE processing errors occurred, THE System SHALL include error details in the summary report
