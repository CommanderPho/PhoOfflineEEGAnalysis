---
name: BaseFileMetadataParser refactor
overview: Generalize BaseFileMetadataParser by removing video-specific behavior and column names, then implement VideoMetadataParser as a thin subclass that delegates to the base while preserving the existing video-named public API used across the codebase.
todos: []
isProject: false
---

# BaseFileMetadataParser refactor and VideoMetadataParser subclass

## Current state

- [file_metadata.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\file_metadata.py): `BaseFileMetadataParser` already exists but contains video-specific logic (cv2, video column names, video docstrings).
- [video_metadata.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\video_metadata.py): `VideoMetadataParser` subclasses it but duplicates almost all logic and exposes the names used by callers: `parse_video_folder`, `video_extensions`, `extract_video_metadata`, `get_file_metadata(video_path)`, `is_video_changed`.

**Call sites to preserve (pyPhoTimeline):**

- `VideoMetadataParser.parse_video_folder(folder_path, video_extensions=..., use_cache, force_rebuild)`
- `VideoMetadataParser.extract_datetime_from_filename(name)`
- `VideoMetadataParser.extract_video_metadata(path)`
- `VideoMetadataParser.get_file_metadata(path)`
- `VideoMetadataParser.is_video_changed(path, cached_row)` (used only inside `video_metadata.py`)

---

## 1. Video-specific behavior to remove from BaseFileMetadataParser


| Location                    | Current (video-specific)                                                                                                                                                                                                         | Action in base                                                                                                                                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Docstrings                  | "video", "video filename", "video file"                                                                                                                                                                                          | Use generic "file" wording.                                                                                                                                                                                                                                                            |
| `load_cache`                | Hardcodes `'video_start_datetime'`, `'video_end_datetime'` for `pd.to_datetime`                                                                                                                                                  | Parameterize: `load_cache(cache_path, datetime_columns: Optional[List[str]] = None)`. Parse only columns in that list when present.                                                                                                                                                    |
| `extract_file_metadata`     | Uses **cv2.VideoCapture**, returns `video_num_frames`, `video_fps`, `video_width`, `video_height`, `video_duration`, `video_file_size`                                                                                           | Remove cv2 usage. Make this the extension point: base implementation returns `None`; subclasses override to provide type-specific metadata.                                                                                                                                            |
| `parse_filesystem_folder`   | Default `included_file_extensions` includes `.xdf`; cache file `_video_metadata_cache.csv`; column names `video_file_path`, `video_start_datetime`, `video_end_datetime`; row built from `extract_file_metadata` with fixed keys | Generalize: add parameters for cache filename, path column, start/end datetime column names, and the key in the metadata dict used for duration (so base can compute end from start + duration). No cv2; call `cls.extract_file_metadata(file_path)` and merge returned dict into row. |
| `is_file_changed` docstring | Arg described as "video_path"                                                                                                                                                                                                    | Use "file_path" in base.                                                                                                                                                                                                                                                               |


**Base must not:** import cv2, reference `video_*` column names, or implement video extraction. It should only provide: datetime-from-filename parsing, file stat metadata, cache load/save with configurable datetime columns, file-changed check, and a parameterized folder-parsing loop that uses an overridable `extract_file_metadata`.

---

## 2. BaseFileMetadataParser (file_metadata.py) – target shape

- **extract_datetime_from_filename(filename)** – unchanged (already generic).
- **get_file_metadata(file_path)** – unchanged; docstring generic.
- **is_file_changed(file_path, cached_row)** – unchanged; docstring use `file_path`.
- **load_cache(cache_path, datetime_columns=None)** – read CSV; if `datetime_columns` is provided, for each column in that list that exists in the DataFrame, run `pd.to_datetime` on it; return DataFrame.
- **save_cache(df, cache_path)** – unchanged.
- **extract_file_metadata(file_path)** – base implementation returns `None`. Subclasses override to return a dict (e.g. with a duration key and any other fields). No cv2 in base.
- **parse_filesystem_folder(cls, folder_path, included_file_extensions, use_cache, force_rebuild, cache_filename="_metadata_cache.csv", path_column="file_path", start_datetime_column="start_datetime", end_datetime_column="end_datetime", duration_metadata_key="duration")** – implement the full loop: resolve cache path from `folder_path` and `cache_filename`, load cache via `load_cache(cache_path, datetime_columns=[start_datetime_column, end_datetime_column])`, glob by `included_file_extensions`, use `path_column` for cached lookup and in result rows, use `is_file_changed` for cache reuse, get start datetime from filename via `extract_datetime_from_filename`, get metadata via `cls.extract_file_metadata(file_path)`, compute end from start + `timedelta(seconds=metadata.get(duration_metadata_key, 0))`, build row as `{path_column: path, start_datetime_column: start, end_datetime_column: end, **metadata, 'cache_file_size': ..., 'cache_file_mtime': ...}`, sort by `start_datetime_column`, save cache. Remove cv2 and all video-named defaults from this method.

Drop the `cv2` import from [file_metadata.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\file_metadata.py).

---

## 3. VideoMetadataParser (video_metadata.py) – thin wrapper

Keep the **existing public names and kwargs** used by callers; implement by delegation and one real override.

- **extract_datetime_from_filename** – do not override (inherit base); or one-line delegate to `super().extract_datetime_from_filename` if docstring should stay video-oriented.
- **get_file_metadata(video_path)** – keep signature; body: `return super().get_file_metadata(video_path)` (param name `video_path` preserved).
- **is_video_changed(video_path, cached_row)** – keep name; body: `return super().is_file_changed(video_path, cached_row)`.
- **load_cache(cache_path)** – override to call `super().load_cache(cache_path, datetime_columns=['video_start_datetime', 'video_end_datetime'])` so cache behavior is unchanged.
- **save_cache** – no override (use base).
- **extract_video_metadata(video_path)** – keep as the only place that uses **cv2** and returns the video dict (`video_num_frames`, `video_fps`, `video_width`, `video_height`, `video_duration`, `video_file_size`). This is the video-specific implementation.
- **extract_file_metadata(file_path)** – override to call `cls.extract_video_metadata(file_path)` so the base’s `parse_filesystem_folder` can use it without knowing about video.
- **parse_video_folder(cls, folder_path, video_extensions=[...], use_cache=True, force_rebuild=False)** – replace current large body with a single call to the base using video-specific parameters, e.g.:
`return cls.parse_filesystem_folder(folder_path, included_file_extensions=video_extensions, use_cache=use_cache, force_rebuild=force_rebuild, cache_filename="_video_metadata_cache.csv", path_column="video_file_path", start_datetime_column="video_start_datetime", end_datetime_column="video_end_datetime", duration_metadata_key="video_duration")`
(Exact parameter names and defaults to match base’s new signature; keep `video_extensions` as the public kwarg.)

No duplication of the folder-parsing or cache logic; all of that lives in the base. Video-specific behavior is confined to: cv2 in `extract_video_metadata`, video column names and cache filename in the `parse_video_folder` call, and the video-named wrappers above.

---

## 4. Summary

- **Base:** Generic file metadata parser: datetime from filename, file stat, cache with configurable datetime columns, file-changed check, parameterized `parse_filesystem_folder` using overridable `extract_file_metadata` (default `None`). No video terminology and no cv2.
- **Video:** Subclass overrides `extract_file_metadata` to call `extract_video_metadata` (cv2); wraps base methods with video-named APIs (`get_file_metadata(video_path)`, `is_video_changed`, `load_cache` with video datetime columns); `parse_video_folder` is a thin call to `parse_filesystem_folder` with video column names and `_video_metadata_cache.csv`. All existing call sites remain valid with no changes.

