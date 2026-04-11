"""
SignAI - Translation Module
English to Tamil/Hindi translation
"""

from collections import OrderedDict

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Warning: googletrans not available. Using fallback translations.")


class Translator:
    """
    Translation service for converting English text to Indian languages
    """
    
    def __init__(self, cache_max: int = 512):
        """Initialize translator"""
        if GOOGLETRANS_AVAILABLE:
            self.translator = GoogleTranslator()
        else:
            self.translator = None
        self._cache_max = cache_max
        self._cache: "OrderedDict[tuple, str]" = OrderedDict()
        
        # Fallback dictionary for common gestures
        self.fallback_translations = {
            'HELLO': {
                'tamil': 'வணக்கம்',
                'hindi': 'नमस्ते'
            },
            'THANK YOU': {
                'tamil': 'நன்றி',
                'hindi': 'धन्यवाद'
            },
            'HELP': {
                'tamil': 'உதவி',
                'hindi': 'मदद'
            },
            'YES': {
                'tamil': 'ஆம்',
                'hindi': 'हाँ'
            },
            'NO': {
                'tamil': 'இல்லை',
                'hindi': 'नहीं'
            },
            'PLEASE': {
                'tamil': 'தயவுசெய்து',
                'hindi': 'कृपया'
            },
            'SORRY': {
                'tamil': 'மன்னிக்கவும்',
                'hindi': 'माफ़ कीजिए'
            },
            'SOS': {
                'tamil': 'அவசரம்',
                'hindi': 'आपातकाल'
            },
            'DANGER': {
                'tamil': 'ஆபத்து',
                'hindi': 'खतरा'
            },
            'GOOD': {
                'tamil': 'நல்லது',
                'hindi': 'अच्छा'
            },
            'BAD': {
                'tamil': 'மோசமான',
                'hindi': 'बुरा'
            },
            'WATER': {
                'tamil': 'நீர்',
                'hindi': 'पानी'
            },
            'FOOD': {
                'tamil': 'உணவு',
                'hindi': 'भोजन'
            },
            'TOILET': {
                'tamil': 'கழிவறை',
                'hindi': 'शौचालय'
            },
            'UNKNOWN': {
                'tamil': 'தெரியாது',
                'hindi': 'अज्ञात'
            }
        }
    
    def translate_to_tamil(self, text):
        """
        Translate English text to Tamil
        
        Args:
            text: English text to translate
            
        Returns:
            translated_text: Tamil translation
        """
        return self._translate(text, 'tamil')
    
    def translate_to_hindi(self, text):
        """
        Translate English text to Hindi
        
        Args:
            text: English text to translate
            
        Returns:
            translated_text: Hindi translation
        """
        return self._translate(text, 'hindi')
    
    def _translate(self, text, target_language):
        """
        Internal translation method
        
        Args:
            text: Text to translate
            target_language: 'tamil' or 'hindi'
            
        Returns:
            translated_text: Translated string
        """
        if not text:
            return ""
        
        text = text.strip().upper()
        cache_key = (text, target_language)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        # Check fallback dictionary first
        if text in self.fallback_translations:
            out = self.fallback_translations[text][target_language]
            self._cache_store(cache_key, out)
            return out
        
        # Try Google Translate
        if GOOGLETRANS_AVAILABLE and self.translator:
            try:
                lang_code = 'ta' if target_language == 'tamil' else 'hi'
                result = self.translator.translate(text, src='en', dest=lang_code)
                out = result.text
                self._cache_store(cache_key, out)
                return out
            except Exception as e:
                print(f"Translation error: {e}")
                return self._get_fallback_message(target_language)
        else:
            return self._get_fallback_message(target_language)

    def _cache_store(self, key, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
    
    def _get_fallback_message(self, language):
        """Return fallback message when translation fails"""
        if language == 'tamil':
            return "மொழிபெயர்ப்பு இல்லை"
        else:
            return "अनुवाद उपलब्ध नहीं"
    
    def translate_both(self, text):
        """
        Translate to both Tamil and Hindi
        
        Args:
            text: English text to translate
            
        Returns:
            translations: Dictionary with both translations
        """
        return {
            'english': text,
            'tamil': self.translate_to_tamil(text),
            'hindi': self.translate_to_hindi(text)
        }
    
    def batch_translate(self, texts, target_language):
        """
        Translate multiple texts
        
        Args:
            texts: List of English texts
            target_language: 'tamil' or 'hindi'
            
        Returns:
            translations: List of translated texts
        """
        translations = []
        for text in texts:
            translations.append(self._translate(text, target_language))
        return translations
    
    def add_custom_translation(self, english_text, tamil_text, hindi_text):
        """
        Add custom translation to fallback dictionary
        
        Args:
            english_text: English phrase
            tamil_text: Tamil translation
            hindi_text: Hindi translation
        """
        key = english_text.upper()
        self.fallback_translations[key] = {
            'tamil': tamil_text,
            'hindi': hindi_text
        }
    
    def get_all_gestures(self):
        """
        Get all available gesture translations
        
        Returns:
            gestures: Dictionary of all gestures with translations
        """
        return self.fallback_translations.copy()
    
    def is_translation_available(self):
        """
        Check if translation service is available
        
        Returns:
            available: Boolean indicating if translation works
        """
        return GOOGLETRANS_AVAILABLE and self.translator is not None


# Utility functions for testing
def test_translator():
    """Test translation functionality"""
    print("Testing Translator")
    print("=" * 60)
    
    translator = Translator()
    
    # Test common gestures
    test_words = ['HELLO', 'THANK YOU', 'HELP', 'YES', 'NO']
    
    for word in test_words:
        translations = translator.translate_both(word)
        print(f"\nEnglish: {translations['english']}")
        print(f"Tamil  : {translations['tamil']}")
        print(f"Hindi  : {translations['hindi']}")
    
    print("\n" + "=" * 60)
    print(f"Translation available: {translator.is_translation_available()}")
    print(f"Total gestures in dictionary: {len(translator.fallback_translations)}")
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_translator()