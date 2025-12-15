"""
Track registry for extensible track registration.

This module provides a registry system for mapping stream names/types to track classes,
allowing new track types to be registered dynamically.
"""

from typing import Dict, Tuple, Optional, Type, List
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.EEGRecordingTrack import EEGRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.MotionRecordingTrack import MotionRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.PhoLogTrack import PhoLogTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.XDFStreamTrack import XDFStreamTrack


class TrackRegistry:
    """
    Registry for mapping stream names and types to track classes.
    
    Allows dynamic registration of new track types for extensibility.
    """
    
    def __init__(self):
        # Map stream names to (display_name, track_class) tuples
        self._stream_name_to_track: Dict[str, Tuple[str, Type[TrackWidget]]] = {}
        
        # Map stream types to (display_name, track_class) tuples
        self._type_to_track: Dict[str, Tuple[str, Type[TrackWidget]]] = {}
        
        # Register default mappings
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default stream name and type mappings."""
        # Stream name mappings
        self.register_stream_name('Epoc X', ('EEG', EEGRecordingTrack))
        self.register_stream_name('Epoc X Motion', ('Motion', MotionRecordingTrack))
        self.register_stream_name('TextLogger', ('PHO_LOG', PhoLogTrack))
        self.register_stream_name('EventBoard', ('PHO_LOG', PhoLogTrack))
        self.register_stream_name('Epoc X eQuality', ('EEG Quality', EEGRecordingTrack))
        
        # Type mappings
        self.register_type('EEG', ('EEG', EEGRecordingTrack))
        self.register_type('SIGNAL', ('Motion', MotionRecordingTrack))
        self.register_type('Markers', ('PHO_LOG', PhoLogTrack))
        self.register_type('Raw', ('EEG', EEGRecordingTrack))
    
    def register_stream_name(self, stream_name: str, track_info: Tuple[str, Type[TrackWidget]]):
        """
        Register a stream name mapping.
        
        Args:
            stream_name: The stream name to map
            track_info: Tuple of (display_name, track_class)
        """
        self._stream_name_to_track[stream_name] = track_info
    
    def register_type(self, stream_type: str, track_info: Tuple[str, Type[TrackWidget]]):
        """
        Register a stream type mapping.
        
        Args:
            stream_type: The stream type to map
            track_info: Tuple of (display_name, track_class)
        """
        self._type_to_track[stream_type] = track_info
    
    def get_track_for_stream(self, stream_name: Optional[str] = None, stream_type: Optional[str] = None) -> Optional[Tuple[str, Type[TrackWidget]]]:
        """
        Get track class for a stream, checking name first, then type.
        
        Args:
            stream_name: Optional stream name
            stream_type: Optional stream type
        
        Returns:
            Tuple of (display_name, track_class) or None if no match found
        """
        # Try stream name first
        if stream_name and stream_name in self._stream_name_to_track:
            return self._stream_name_to_track[stream_name]
        
        # Fall back to type
        if stream_type and stream_type in self._type_to_track:
            return self._type_to_track[stream_type]
        
        return None
    
    def get_all_stream_names(self) -> List[str]:
        """Get all registered stream names."""
        return list(self._stream_name_to_track.keys())
    
    def get_all_types(self) -> List[str]:
        """Get all registered stream types."""
        return list(self._type_to_track.keys())


# Global registry instance
_default_registry = TrackRegistry()


def get_default_registry() -> TrackRegistry:
    """Get the default track registry instance."""
    return _default_registry


def register_track(stream_name: Optional[str] = None, stream_type: Optional[str] = None, 
                   display_name: str = None, track_class: Type[TrackWidget] = None):
    """
    Convenience function to register a track in the default registry.
    
    Args:
        stream_name: Optional stream name to register
        stream_type: Optional stream type to register
        display_name: Display name for the track
        track_class: Track class to use
    """
    if display_name is None or track_class is None:
        raise ValueError("display_name and track_class are required")
    
    registry = get_default_registry()
    track_info = (display_name, track_class)
    
    if stream_name:
        registry.register_stream_name(stream_name, track_info)
    if stream_type:
        registry.register_type(stream_type, track_info)

