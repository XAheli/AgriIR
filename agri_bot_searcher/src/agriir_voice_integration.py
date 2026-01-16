#!/usr/bin/env python3
"""
AgriIR Voice Transcription Integration Module

This module integrates the agri_bot voice transcription functionality
into the agri_bot_searcher system with proper language support.
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    # Import SarvamAI for voice transcription
    from sarvamai import SarvamAI
    import subprocess
    import tempfile
    import re
    HAS_SARVAM = True
except ImportError:
    HAS_SARVAM = False
    logging.warning("SarvamAI not available. Voice transcription disabled.")

# Disable agri_bot local models due to IndicTrans dependency conflicts
HAS_AGRI_BOT = False
logging.info("Local voice models disabled due to IndicTrans dependency conflicts. Using SarvamAI for voice transcription.")

# Implement essential functions for SarvamAI voice transcription
def mono_channel(audio_path):
    """Convert audio to mono channel using ffmpeg"""
    try:
        web_audio_path = "temp_audio_mono.wav"
        command = ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", web_audio_path]
        result = subprocess.run(command, capture_output=True, check=True)
        return web_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Audio conversion failed: {e}")
        return audio_path  # Return original if conversion fails
    except Exception as e:
        logging.error(f"Audio processing error: {e}")
        return audio_path

def speech_to_text(audio_path, sarvam_api):
    """Convert speech to text using SarvamAI"""
    if not HAS_SARVAM or not sarvam_api:
        raise Exception("SarvamAI not available or API key not provided")
    
    try:
        client = SarvamAI(api_subscription_key=sarvam_api)
        
        # Read the audio file
        with open(audio_path, "rb") as audio_file:
            response = client.speech_to_text.translate(
                file=audio_file,
                model="saaras:v2.5"
            )
        
        # Extract transcript from response - handle different response formats
        transcript_text = ""
        
        if hasattr(response, 'transcript'):
            transcript_text = response.transcript
        elif isinstance(response, dict):
            transcript_text = response.get('transcript', '')
        else:
            # Try regex pattern matching as fallback
            response_str = str(response)
            pattern = r"transcript['\"]?\s*[:=]\s*['\"]([^'\"]*)['\"]"
            match = re.search(pattern, response_str, re.IGNORECASE)
            if match:
                transcript_text = match.group(1)
            else:
                # Another pattern for SarvamAI response
                pattern = r"transcript='(.*?)'"
                match = re.search(pattern, response_str, re.DOTALL)
                if match:
                    transcript_text = match.group(1)
                else:
                    transcript_text = response_str
        
        # Clean up the transcript
        if transcript_text:
            transcript_text = transcript_text.strip()
            # Remove common artifacts
            transcript_text = re.sub(r'^transcript[:\s=]*', '', transcript_text, flags=re.IGNORECASE)
            transcript_text = re.sub(r'language_code.*$', '', transcript_text, flags=re.IGNORECASE)
            transcript_text = transcript_text.strip()
        
        if not transcript_text:
            transcript_text = "No speech detected in audio"
            
        logging.info(f"SarvamAI transcription successful: {transcript_text[:100]}...")
        return transcript_text
                
    except Exception as e:
        logging.error(f"SarvamAI transcription error: {e}")
        raise Exception(f"SarvamAI transcription failed: {str(e)}")

def text_to_text(text, sarvam_api, src_lan, tg_lan="en-IN"):
    """Translate text using SarvamAI"""
    if not HAS_SARVAM or not sarvam_api:
        raise Exception("SarvamAI not available or API key not provided")
    
    try:
        client = SarvamAI(api_subscription_key=sarvam_api)
        response = client.text.translate(
            input=text,
            source_language_code=src_lan,  
            target_language_code=tg_lan,
            model="sarvam-translate:v1"
        )
        
        # Extract translated text from response - handle different response formats
        translated_text = ""
        
        if hasattr(response, 'translated_text'):
            translated_text = response.translated_text
        elif isinstance(response, dict):
            translated_text = response.get('translated_text', '')
        else:
            # Try regex pattern matching as fallback
            response_str = str(response)
            pattern = r"translated_text['\"]?\s*[:=]\s*['\"]([^'\"]*)['\"]"
            match = re.search(pattern, response_str, re.IGNORECASE)
            if match:
                translated_text = match.group(1)
            else:
                # Another pattern for SarvamAI response
                pattern = r"translated_text='(.*?)'"
                match = re.search(pattern, response_str, re.DOTALL)
                if match:
                    translated_text = match.group(1)
                else:
                    translated_text = response_str
        
        # Clean up the translation
        if translated_text:
            translated_text = translated_text.strip()
            # Remove common artifacts
            translated_text = re.sub(r'^translated_text[:\s=]*', '', translated_text, flags=re.IGNORECASE)
            translated_text = re.sub(r'source_language_code.*$', '', translated_text, flags=re.IGNORECASE)
            translated_text = translated_text.strip()
        
        if not translated_text:
            # Fallback: return original text if translation fails
            translated_text = text
            
        logging.info(f"SarvamAI translation successful: {text[:50]}... -> {translated_text[:50]}...")
        return translated_text
                
    except Exception as e:
        logging.error(f"SarvamAI translation error: {e}")
        # Fallback: return original text if translation fails
        logging.warning(f"Translation failed, returning original text: {text}")
        return text


class AgriIRVoiceTranscriber:
    """
    Enhanced voice transcriber for AgriIR using agri_bot functionality
    """
    
    # Language mappings for AgriIR
    LANGUAGE_MAPPINGS = {
        'asm_Beng': {'name': 'Assamese (Bengali script)', 'code': 'asm_Beng'},
        'ben_Beng': {'name': 'Bengali', 'code': 'ben_Beng'},
        'brx_Deva': {'name': 'Bodo', 'code': 'brx_Deva'},
        'doi_Deva': {'name': 'Dogri', 'code': 'doi_Deva'},
        'guj_Gujr': {'name': 'Gujarati', 'code': 'guj_Gujr'},
        'hin_Deva': {'name': 'Hindi', 'code': 'hin_Deva'},
        'kan_Knda': {'name': 'Kannada', 'code': 'kan_Knda'},
        'gom_Deva': {'name': 'Konkani', 'code': 'gom_Deva'},
        'kas_Arab': {'name': 'Kashmiri (Arabic script)', 'code': 'kas_Arab'},
        'kas_Deva': {'name': 'Kashmiri (Devanagari script)', 'code': 'kas_Deva'},
        'mai_Deva': {'name': 'Maithili', 'code': 'mai_Deva'},
        'mal_Mlym': {'name': 'Malayalam', 'code': 'mal_Mlym'},
        'mni_Beng': {'name': 'Manipuri (Bengali script)', 'code': 'mni_Beng'},
        'mni_Mtei': {'name': 'Manipuri (Meitei script)', 'code': 'mni_Mtei'},
        'mar_Deva': {'name': 'Marathi', 'code': 'mar_Deva'},
        'npi_Deva': {'name': 'Nepali', 'code': 'npi_Deva'},
        'ory_Orya': {'name': 'Odia', 'code': 'ory_Orya'},
        'pan_Guru': {'name': 'Punjabi', 'code': 'pan_Guru'},
        'san_Deva': {'name': 'Sanskrit', 'code': 'san_Deva'},
        'sat_Olck': {'name': 'Santali (Ol Chiki script)', 'code': 'sat_Olck'},
        'snd_Arab': {'name': 'Sindhi (Arabic script)', 'code': 'snd_Arab'},
        'snd_Deva': {'name': 'Sindhi (Devanagari script)', 'code': 'snd_Deva'},
        'urd_Arab': {'name': 'Urdu', 'code': 'urd_Arab'},
        'eng_Latn': {'name': 'English (Latin script)', 'code': 'eng_Latn'}
    }
    
    def __init__(self):
        """Initialize the AgriIR voice transcriber"""
        self.agri_bot_available = HAS_AGRI_BOT
        self.sarvam_available = HAS_SARVAM
        self._models_loaded = False
        self._ai_bharat_model = None
        self._indic_model = None
        self._indic_tokenizer = None
        
        if not self.sarvam_available:
            logging.warning("SarvamAI not available. Voice transcription requires API key.")
        if not self.agri_bot_available:
            logging.info("Local models disabled due to dependency conflicts. Using SarvamAI for voice transcription.")
    
    def _ensure_models_loaded(self, use_local_model: bool = False, hf_token: Optional[str] = None):
        """Ensure models are loaded if using local models (disabled by default due to IndicTrans conflicts)"""
        if not self.agri_bot_available or not use_local_model:
            return
            
        if self._models_loaded:
            return
            
        try:
            # NOTE: IndicTrans models disabled due to dependency conflicts
            # Using SarvamAI API as primary transcription method
            logging.warning("Local model loading disabled due to IndicTrans dependency conflicts")
            logging.info("Using SarvamAI API for voice transcription instead")
            return
            
            # # Login to Hugging Face if token provided
            # if hf_token:
            #     login_in(hf_token)
            
            # # Load AI Bharat model
            # logging.info("Loading AI Bharat model...")
            # self._ai_bharat_model = ai_bharat()
            
            # # Load IndicTrans model (DISABLED)
            # # logging.info("Loading IndicTrans model...")
            # # self._indic_model, self._indic_tokenizer = load_indic_trans()
            
            # self._models_loaded = True
            # logging.info("Models loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def _prepare_audio_file(self, audio_path: str) -> str:
        """
        Prepare audio file for transcription (convert to mono, 16kHz)
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the processed audio file
        """
        try:
            # Use agri_bot's mono_channel function
            processed_path = mono_channel(audio_path)
            return processed_path
        except Exception as e:
            logging.error(f"Error processing audio file: {e}")
            # Fallback: try manual conversion
            return self._manual_audio_conversion(audio_path)
    
    def _manual_audio_conversion(self, audio_path: str) -> str:
        """
        Manual audio conversion using ffmpeg
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the processed audio file
        """
        output_path = tempfile.mktemp(suffix='_mono.wav')
        
        try:
            command = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ac', '1',  # mono channel
                '-ar', '16000',  # 16kHz sample rate
                '-f', 'wav',  # WAV format
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                
            return output_path
            
        except Exception as e:
            logging.error(f"Manual audio conversion failed: {e}")
            # If conversion fails, return original path and hope for the best
            return audio_path
    
    def transcribe_audio(self, 
                        audio_path: str,
                        language_code: str = 'hin_Deva',
                        use_local_model: bool = False,  # Default to SarvamAI due to IndicTrans conflicts
                        api_key: Optional[str] = None,
                        hf_token: Optional[str] = None) -> Tuple[str, str]:
        """
        Transcribe audio file and return both original and English text
        
        Args:
            audio_path: Path to the audio file
            language_code: Language code (e.g., 'hin_Deva', 'mar_Deva')
            use_local_model: Whether to use local models or SarvamAI (SarvamAI recommended)
            api_key: SarvamAI API key (required for external transcription)
            hf_token: Hugging Face token (if using local models)
            
        Returns:
            Tuple of (original_text, english_text)
        """
        try:
            # Validate language code
            if language_code not in self.LANGUAGE_MAPPINGS:
                logging.warning(f"Unknown language code: {language_code}, using Hindi as fallback")
                language_code = 'hin_Deva'
            
            # Prepare audio file
            processed_audio_path = self._prepare_audio_file(audio_path)
            
            try:
                if use_local_model and self.agri_bot_available:
                    # Local models currently disabled due to IndicTrans dependency conflicts
                    logging.warning("Local models disabled. Falling back to SarvamAI...")
                    use_local_model = False
                    
                    # # Use local models (DISABLED)
                    # self._ensure_models_loaded(use_local_model, hf_token)
                    
                    # # Transcribe using AI Bharat
                    # original_text = speech_to_text_bharat(
                    #     model=self._ai_bharat_model,
                    #     audio_path=processed_audio_path
                    # )
                    
                    # # Translate to English using IndicTrans
                    # english_text = translate_indic(
                    #     model=self._indic_model,
                    #     tokenizer=self._indic_tokenizer,
                    #     text=[original_text],
                    #     audio_code=language_code
                    # )[0]
                
                # Use SarvamAI (recommended approach)
                if not api_key:
                    return ("SarvamAI API key required for voice transcription. " +
                           "Click the help icon to learn how to get one."), \
                           ("SarvamAI API key required for voice transcription. " +
                           "Click the help icon to learn how to get one.")
                
                try:
                    # Step 1: Transcribe speech to text (in original language)
                    original_text = speech_to_text(
                        audio_path=processed_audio_path,
                        sarvam_api=api_key
                    )
                    
                    # Step 2: Translate to English if not already in English
                    if language_code == 'eng_Latn':
                        # Already English, no translation needed
                        english_text = original_text
                    else:
                        try:
                            # Map language codes to SarvamAI format
                            src_language = self._map_to_sarvam_language(language_code)
                            
                            english_text = text_to_text(
                                text=original_text,
                                sarvam_api=api_key,
                                src_lan=src_language,
                                tg_lan="en-IN"
                            )
                        except Exception as trans_error:
                            logging.warning(f"Translation failed: {trans_error}, using original text")
                            english_text = original_text
                    
                    logging.info(f"SarvamAI transcription successful for language: {language_code}")
                    logging.info(f"Original: {original_text[:100]}...")
                    logging.info(f"English: {english_text[:100]}...")
                    
                except Exception as sarvam_error:
                    error_msg = f"SarvamAI transcription failed: {str(sarvam_error)}"
                    logging.error(error_msg)
                    return error_msg, error_msg
                
                return original_text, english_text
                
            finally:
                # Clean up processed audio file if it's different from original
                if processed_audio_path != audio_path:
                    try:
                        os.unlink(processed_audio_path)
                    except OSError:
                        pass
                        
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            logging.error(error_msg)
            return error_msg, error_msg
    
    def transcribe_from_blob(self,
                           audio_blob: bytes,
                           language_code: str = 'hin_Deva',
                           use_local_model: bool = True,
                           api_key: Optional[str] = None,
                           hf_token: Optional[str] = None) -> Tuple[str, str]:
        """
        Transcribe audio from binary blob
        
        Args:
            audio_blob: Audio data as bytes
            language_code: Language code
            use_local_model: Whether to use local models
            api_key: SarvamAI API key
            hf_token: Hugging Face token
            
        Returns:
            Tuple of (original_text, english_text)
        """
        # Save blob to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_blob)
            temp_path = tmp_file.name
        
        try:
            return self.transcribe_audio(
                audio_path=temp_path,
                language_code=language_code,
                use_local_model=use_local_model,
                api_key=api_key,
                hf_token=hf_token
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of supported languages"""
        return self.LANGUAGE_MAPPINGS.copy()
    
    def _map_to_sarvam_language(self, language_code: str) -> str:
        """Map AgriIR language codes to SarvamAI format"""
        # Mapping from AgriIR codes to SarvamAI language codes
        mapping = {
            'hin_Deva': 'hi-IN',      # Hindi
            'ben_Beng': 'bn-IN',      # Bengali
            'guj_Gujr': 'gu-IN',      # Gujarati
            'kan_Knda': 'kn-IN',      # Kannada
            'mal_Mlym': 'ml-IN',      # Malayalam
            'mar_Deva': 'mr-IN',      # Marathi
            'npi_Deva': 'ne-NP',      # Nepali
            'ory_Orya': 'or-IN',      # Odia
            'pan_Guru': 'pa-IN',      # Punjabi
            'urd_Arab': 'ur-PK',      # Urdu
            'eng_Latn': 'en-IN',      # English
            # Add more mappings as needed
        }
        
        return mapping.get(language_code, 'hi-IN')  # Default to Hindi if not found
    
    def is_available(self) -> bool:
        """Check if voice transcription is available"""
        return HAS_SARVAM  # Voice available if SarvamAI is available
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable language name from code"""
        return self.LANGUAGE_MAPPINGS.get(language_code, {}).get('name', 'Unknown Language')


# Create a global instance for easy import
agriir_transcriber = AgriIRVoiceTranscriber()


def transcribe_audio_file(audio_path: str,
                         language_code: str = 'hin_Deva',
                         use_local_model: bool = True,
                         api_key: Optional[str] = None,
                         hf_token: Optional[str] = None) -> Tuple[str, str]:
    """
    Convenience function for transcribing audio files
    
    Args:
        audio_path: Path to the audio file
        language_code: Language code (e.g., 'hin_Deva')
        use_local_model: Whether to use local models
        api_key: SarvamAI API key
        hf_token: Hugging Face token
        
    Returns:
        Tuple of (original_text, english_text)
    """
    return agriir_transcriber.transcribe_audio(
        audio_path=audio_path,
        language_code=language_code,
        use_local_model=use_local_model,
        api_key=api_key,
        hf_token=hf_token
    )


def get_supported_languages() -> Dict[str, Dict[str, str]]:
    """Get dictionary of supported languages"""
    return agriir_transcriber.get_supported_languages()


if __name__ == '__main__':
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AgriIR Voice Transcription')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--language', default='hin_Deva', help='Language code')
    parser.add_argument('--use-sarvam', action='store_true', help='Use SarvamAI instead of local models')
    parser.add_argument('--api-key', help='SarvamAI API key')
    parser.add_argument('--hf-token', help='Hugging Face token')
    
    args = parser.parse_args()
    
    transcriber = AgriIRVoiceTranscriber()
    
    if not transcriber.is_available():
        print("Voice transcription not available - agri_bot modules not found")
        sys.exit(1)
    
    print(f"Transcribing audio: {args.audio}")
    print(f"Language: {transcriber.get_language_name(args.language)}")
    print(f"Using: {'SarvamAI' if args.use_sarvam else 'Local models'}")
    
    original, english = transcriber.transcribe_audio(
        audio_path=args.audio,
        language_code=args.language,
        use_local_model=not args.use_sarvam,
        api_key=args.api_key,
        hf_token=args.hf_token
    )
    
    print(f"\nOriginal: {original}")
    print(f"English: {english}")
