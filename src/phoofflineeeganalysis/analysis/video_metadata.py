import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import pandas as pd
from attrs import define, field


@define(slots=False)
class VideoMetadataParser:
    """
    Parses video folders and extracts metadata including datetime from filenames.
    
    Usage:
        from phoofflineeeganalysis.analysis.video_metadata import VideoMetadataParser
        
        folder_path = Path(r"M:\\ScreenRecordings\\EyeTrackerVR_Recordings")
        df = VideoMetadataParser.parse_video_folder(folder_path)
        print(df)
    """
    
    @classmethod
    def extract_datetime_from_filename(cls, filename: str) -> Optional[datetime]:
        """
        Extract datetime from video filename.
        
        Examples:
            'Debut_2025-07-03T230155.mp4' -> datetime(2025, 7, 3, 23, 1, 55)
            'Video_2025-12-25T120000.avi' -> datetime(2025, 12, 25, 12, 0, 0)
        """
        # Pattern to match: Debut_2025-07-03T230155 or similar
        candidates = re.findall(r'\d{4}[-_]?\d{2}[-_]?\d{2}[ T_-]?\d{2}[:\-]?\d{2}[:\-]?\d{2}', filename)
        for cand in candidates:
            normalized = cand.replace("_", "T").replace(" ", "T")
            for fmt in [
                "%Y-%m-%dT%H-%M-%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H%M%S",
                "%Y%m%dT%H%M%S",
                "%Y%m%d_%H%M%S",
                "%Y%m%d-%H%M%S",
                "%Y%m%d%H%M%S"
            ]:
                try:
                    return datetime.strptime(normalized, fmt)
                except ValueError:
                    continue
        return None
    
    @classmethod
    def get_file_metadata(cls, video_path: Path) -> Dict[str, Any]:
        """
        Extract file size and modification time for cache validation.
        
        Returns:
            Dictionary with file_size and file_mtime.
        """
        try:
            stat = video_path.stat()
            return {
                'file_size': stat.st_size,
                'file_mtime': stat.st_mtime,
            }
        except Exception:
            return {'file_size': 0, 'file_mtime': 0.0}
    
    @classmethod
    def is_video_changed(cls, video_path: Path, cached_row: pd.Series) -> bool:
        """
        Check if a video file has been modified since it was cached.
        
        Args:
            video_path: Path to the video file
            cached_row: Cached row from DataFrame with cache_file_size and cache_file_mtime
            
        Returns:
            True if file was modified or doesn't exist, False otherwise.
        """
        if not video_path.exists():
            return True
        
        try:
            current_metadata = cls.get_file_metadata(video_path)
            cached_size = cached_row.get('cache_file_size') if 'cache_file_size' in cached_row.index else 0
            cached_mtime = cached_row.get('cache_file_mtime') if 'cache_file_mtime' in cached_row.index else 0.0
            
            return (current_metadata['file_size'] != cached_size or 
                    abs(current_metadata['file_mtime'] - cached_mtime) > 0.1)  # 0.1 second tolerance
        except Exception:
            return True
    
    @classmethod
    def load_cache(cls, cache_path: Path) -> pd.DataFrame:
        """
        Load cached video metadata from CSV file.
        
        Args:
            cache_path: Path to the cache CSV file
            
        Returns:
            DataFrame with cached metadata, or empty DataFrame if cache doesn't exist or is corrupted.
        """
        if not cache_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(cache_path)
            
            # Parse datetime columns
            if 'video_start_datetime' in df.columns:
                df['video_start_datetime'] = pd.to_datetime(df['video_start_datetime'])
            if 'video_end_datetime' in df.columns:
                df['video_end_datetime'] = pd.to_datetime(df['video_end_datetime'])
            
            return df
        except Exception:
            # Corrupted cache - return empty DataFrame
            return pd.DataFrame()
    
    @classmethod
    def save_cache(cls, df: pd.DataFrame, cache_path: Path) -> None:
        """
        Save video metadata DataFrame to cache CSV file.
        
        Args:
            df: DataFrame with video metadata to cache
            cache_path: Path where cache should be saved
        """
        if df.empty:
            return
        
        try:
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(cache_path, index=False)
        except Exception:
            # Silently fail if we can't save cache
            pass
    
    @classmethod
    def extract_video_metadata(cls, video_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a video file using cv2.VideoCapture.
        
        Returns:
            Dictionary with video metadata or None if extraction fails.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = num_frames / fps if fps > 0 else 0.0
            
            # Get file size
            file_size = video_path.stat().st_size
            
            cap.release()
            
            return {
                'video_num_frames': num_frames,
                'video_fps': fps,
                'video_width': width,
                'video_height': height,
                'video_duration': duration,
                'video_file_size': file_size,
            }
        except Exception:
            return None
    
    @classmethod
    def parse_video_folder(cls, folder_path: Path, video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'], use_cache: bool = True, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Parse all videos in a folder and return a DataFrame with metadata.
        Uses caching to speed up subsequent runs by only processing new or modified videos.
        
        Args:
            folder_path: Path to folder containing videos
            video_extensions: List of video file extensions to process
            use_cache: If True, use cached metadata for unchanged videos (default: True)
            force_rebuild: If True, ignore cache and rebuild from scratch (default: False)
            
        Returns:
            DataFrame with columns:
            - video_start_datetime: Parsed datetime from filename
            - video_duration: Duration in seconds
            - video_end_datetime: Calculated end datetime
            - video_num_frames: Total number of frames
            - video_fps: Frames per second
            - video_width: Video width in pixels
            - video_height: Video height in pixels
            - video_file_path: Full path to video file
            - video_file_size: File size in bytes
            - cache_file_size: File size used for cache validation
            - cache_file_mtime: File modification time used for cache validation
            
        DataFrame is sorted by video_start_datetime.
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return pd.DataFrame()
        
        # Determine cache path
        cache_path = folder_path / "_video_metadata_cache.csv"
        
        # Load existing cache if enabled and not forcing rebuild
        cached_df = pd.DataFrame()
        if use_cache and not force_rebuild:
            cached_df = cls.load_cache(cache_path)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder_path.glob(f"*{ext}"))
            video_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            # No videos found - return empty DataFrame and clear cache if it exists
            if cache_path.exists() and use_cache:
                try:
                    cache_path.unlink()
                except Exception:
                    pass
            return pd.DataFrame()
        
        # Build dictionary of cached entries by file path
        cached_by_path = {}
        if not cached_df.empty and 'video_file_path' in cached_df.columns:
            for idx, row in cached_df.iterrows():
                file_path = row['video_file_path']
                cached_by_path[file_path] = row
        
        # Process each video
        results = []
        processed_paths = set()
        
        for video_path in video_files:
            resolved_path = str(video_path.resolve())
            processed_paths.add(resolved_path)
            
            # Check if we can use cached entry
            use_cached = False
            if use_cache and not force_rebuild and resolved_path in cached_by_path:
                cached_row = cached_by_path[resolved_path]
                if not cls.is_video_changed(video_path, cached_row):
                    # Use cached entry, but update cache metadata
                    cached_entry = cached_row.to_dict()
                    current_metadata = cls.get_file_metadata(video_path)
                    cached_entry['cache_file_size'] = current_metadata['file_size']
                    cached_entry['cache_file_mtime'] = current_metadata['file_mtime']
                    results.append(cached_entry)
                    use_cached = True
            
            if not use_cached:
                # Extract datetime from filename
                video_start_datetime = cls.extract_datetime_from_filename(video_path.name)
                if video_start_datetime is None:
                    continue
                
                # Extract video metadata
                metadata = cls.extract_video_metadata(video_path)
                if metadata is None:
                    continue
                
                # Get file metadata for cache validation
                file_metadata = cls.get_file_metadata(video_path)
                
                # Calculate end datetime
                video_end_datetime = video_start_datetime + timedelta(seconds=metadata['video_duration'])
                
                # Build result row
                result = {
                    'video_start_datetime': video_start_datetime,
                    'video_duration': metadata['video_duration'],
                    'video_end_datetime': video_end_datetime,
                    'video_num_frames': metadata['video_num_frames'],
                    'video_fps': metadata['video_fps'],
                    'video_width': metadata['video_width'],
                    'video_height': metadata['video_height'],
                    'video_file_path': resolved_path,
                    'video_file_size': metadata['video_file_size'],
                    'cache_file_size': file_metadata['file_size'],
                    'cache_file_mtime': file_metadata['file_mtime'],
                }
                results.append(result)
        
        if not results:
            # No valid videos found - clear cache if it exists
            if cache_path.exists() and use_cache:
                try:
                    cache_path.unlink()
                except Exception:
                    pass
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by video_start_datetime
        df = df.sort_values('video_start_datetime').reset_index(drop=True)
        
        # Save updated cache
        if use_cache:
            cls.save_cache(df, cache_path)
        
        return df


if __name__ == "__main__":
    # Example usage
    folder_path = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings")
    
    print(f"Parsing videos in: {folder_path}")
    print("-" * 80)
    
    df = VideoMetadataParser.parse_video_folder(folder_path)
    
    if df.empty:
        print("No videos found or no videos could be parsed.")
    else:
        print(f"\nFound {len(df)} videos:\n")
        print(df.to_string())
        csv_save_path = Path('output').joinpath('2025-12-15_parsed_videos.csv').resolve()
        print(f'csv_save_path: "{csv_save_path.as_posix()}"')
        df.to_csv(csv_save_path)

        ## load with: 
        # csv_save_path = Path('output').joinpath('2025-12-09_parsed_videos.csv').resolve()
        # assert csv_save_path.exists()
        # video_df: pd.DataFrame = pd.read_csv(csv_save_path)

        print("\n" + "-" * 80)
        print("\nSummary statistics:")
        print(f"  Total videos: {len(df)}")
        print(f"  Total duration: {df['video_duration'].sum():.2f} seconds ({df['video_duration'].sum()/3600:.2f} hours)")
        print(f"  Average duration: {df['video_duration'].mean():.2f} seconds")
        print(f"  Date range: {df['video_start_datetime'].min()} to {df['video_end_datetime'].max()}")

