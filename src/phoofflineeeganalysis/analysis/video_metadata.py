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
        
        folder_path = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings")
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
    def parse_video_folder(cls, folder_path: Path, video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']) -> pd.DataFrame:
        """
        Parse all videos in a folder and return a DataFrame with metadata.
        
        Args:
            folder_path: Path to folder containing videos
            video_extensions: List of video file extensions to process
            
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
            
        DataFrame is sorted by video_start_datetime.
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return pd.DataFrame()
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder_path.glob(f"*{ext}"))
            video_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            return pd.DataFrame()
        
        # Process each video
        results = []
        for video_path in video_files:
            # Extract datetime from filename
            video_start_datetime = cls.extract_datetime_from_filename(video_path.name)
            if video_start_datetime is None:
                continue
            
            # Extract video metadata
            metadata = cls.extract_video_metadata(video_path)
            if metadata is None:
                continue
            
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
                'video_file_path': str(video_path.resolve()),
                'video_file_size': metadata['video_file_size'],
            }
            results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by video_start_datetime
        df = df.sort_values('video_start_datetime').reset_index(drop=True)
        
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
        csv_save_path = Path('output').joinpath('2025-12-09_parsed_videos.csv').resolve()
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

