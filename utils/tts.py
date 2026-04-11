"""
SignAI - Text-to-Speech Module
Convert translated text to audio
"""

import os
import hashlib
from pathlib import Path

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. Audio generation disabled.")


class TextToSpeech:
    """
    Text-to-speech conversion for Tamil and Hindi
    """
    
    def __init__(self, cache_dir='static/audio_cache'):
        """
        Initialize TTS engine
        
        Args:
            cache_dir: Directory to cache audio files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.language_codes = {
            'tamil': 'ta',
            'hindi': 'hi',
            'english': 'en'
        }
    
    def generate_speech(self, text, language='tamil', slow=False):
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            language: Language name ('tamil', 'hindi', 'english')
            slow: Speak slowly for better clarity
            
        Returns:
            audio_path: Path to generated audio file (relative to static/)
        """
        if not text or not GTTS_AVAILABLE:
            return None
        
        try:
            # Get language code
            lang_code = self.language_codes.get(language, 'en')
            
            # Generate cache filename based on text and language
            cache_key = self._generate_cache_key(text, language)
            audio_filename = f"{cache_key}.mp3"
            audio_path = self.cache_dir / audio_filename
            
            # Check if already cached
            if audio_path.exists():
                print(f"Using cached audio: {audio_filename}")
                return f"audio_cache/{audio_filename}"
            
            # Generate new audio
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.save(str(audio_path))
            
            print(f"Generated audio: {audio_filename}")
            return f"audio_cache/{audio_filename}"
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def generate_multilingual(self, translations):
        """
        Generate speech for multiple languages
        
        Args:
            translations: Dictionary with 'tamil' and 'hindi' keys
            
        Returns:
            audio_paths: Dictionary with audio file paths
        """
        audio_paths = {}
        
        for lang, text in translations.items():
            if lang in ['tamil', 'hindi', 'english'] and text:
                path = self.generate_speech(text, lang)
                audio_paths[lang] = path
        
        return audio_paths
    
    def _generate_cache_key(self, text, language):
        """Generate unique cache key for text and language"""
        combined = f"{text}_{language}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached audio files"""
        try:
            count = 0
            for file in self.cache_dir.glob("*.mp3"):
                file.unlink()
                count += 1
            print(f"Cleared {count} cached audio files")
            return count
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 0
    
    def get_cache_size(self):
        """
        Get total size of cached audio files
        
        Returns:
            size_mb: Cache size in megabytes
        """
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.mp3"))
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"Error getting cache size: {e}")
            return 0
    
    def is_available(self):
        """Check if TTS is available"""
        return GTTS_AVAILABLE


class OfflineTTS:
    """
    Fallback offline TTS using pyttsx3
    (Less quality but works without internet)
    """
    
    def __init__(self):
        """Initialize offline TTS"""
        self.engine = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
            print("✅ Offline TTS initialized")
        except ImportError:
            print("Warning: pyttsx3 not available")
        except Exception as e:
            print(f"Warning: Could not initialize offline TTS: {e}")
    
    def speak(self, text):
        """
        Speak text immediately (no file generation)
        
        Args:
            text: Text to speak
        """
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error speaking: {e}")
    
    def save_to_file(self, text, filename):
        """
        Save speech to audio file
        
        Args:
            text: Text to convert
            filename: Output filename
        """
        if self.engine:
            try:
                self.engine.save_to_file(text, filename)
                self.engine.runAndWait()
                return filename
            except Exception as e:
                print(f"Error saving audio: {e}")
                return None
        return None


# Combined TTS with fallback
class SmartTTS:
    """
    Smart TTS that tries online first, falls back to offline
    """
    
    def __init__(self):
        """Initialize both TTS engines"""
        self.online_tts = TextToSpeech()
        self.offline_tts = OfflineTTS()
    
    def speak(self, text, language='tamil', use_online=True):
        """
        Generate speech using best available method
        
        Args:
            text: Text to speak
            language: Language
            use_online: Try online TTS first
            
        Returns:
            audio_path: Path to audio file or None
        """
        if use_online and self.online_tts.is_available():
            return self.online_tts.generate_speech(text, language)
        elif self.offline_tts.engine:
            # Use offline for English only
            if language == 'english':
                self.offline_tts.speak(text)
            return None
        else:
            print("No TTS available")
            return None


# Testing function
def test_tts():
    """Test TTS functionality"""
    print("Testing Text-to-Speech")
    print("=" * 60)
    
    tts = TextToSpeech()
    
    if not tts.is_available():
        print("⚠️  gTTS not available - install with: pip install gTTS")
        return
    
    # Test translations
    test_data = {
        'english': 'Hello',
        'tamil': 'வணக்கம்',
        'hindi': 'नमस्ते'
    }
    
    print("\nGenerating audio files...")
    audio_paths = tts.generate_multilingual(test_data)
    
    for lang, path in audio_paths.items():
        if path:
            print(f"✅ {lang.capitalize()}: {path}")
        else:
            print(f"❌ {lang.capitalize()}: Failed")
    
    # Cache info
    print(f"\nCache size: {tts.get_cache_size():.2f} MB")
    print(f"Cache directory: {tts.cache_dir}")
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_tts()