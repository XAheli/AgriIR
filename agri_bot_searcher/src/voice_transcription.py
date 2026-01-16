#!/usr/bin/env python3
"""
Voice Transcription Module for Agriculture Bot Searcher

This module provides voice transcription capabilities for Indian languages
using AI4Bharat models (Conformer) and IndicTrans2 for translation to English.

Features:
- Indian language speech-to-text using AI4Bharat Conformer models
- Translation from Indian languages to English using IndicTrans2
- Support for multiple Indian languages
- Integration with SarvamAI for advanced transcription
- Fallback mechanisms for robust performance

Dependencies:
- nemo_toolkit[asr]
- transformers
- torch
- IndicTransToolkit
- sarvamai (optional)
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import re
from dotenv import load_dotenv

# Disable Torch compile & inductor globally
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

try:
    import nemo.collections.asr as nemo_asr
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from IndicTransToolkit.processor import IndicProcessor
    HAS_NEMO = True
except ImportError as e:
    HAS_NEMO = False
    logging.warning(f"NeMo or related dependencies not available: {e}")

try:
    from sarvamai import SarvamAI
    HAS_SARVAM = True
except ImportError:
    HAS_SARVAM = False
    logging.warning("SarvamAI not available. Install with: pip install sarvamai")


class VoiceTranscriptionError(Exception):
    """Custom exception for voice transcription errors"""
    pass


class VoiceTranscriber:
    """
    Voice transcription handler for Indian languages with English translation
    """
    
    # Supported language mappings
    SUPPORTED_LANGUAGES = {
        'hi': {'name': 'Hindi', 'nemo_id': 'hi', 'indic_code': 'hin_Deva'},
        'mr': {'name': 'Marathi', 'nemo_id': 'mr', 'indic_code': 'mar_Deva'},
        'bn': {'name': 'Bengali', 'nemo_id': 'bn', 'indic_code': 'ben_Beng'},
        'te': {'name': 'Telugu', 'nemo_id': 'te', 'indic_code': 'tel_Telu'},
        'ta': {'name': 'Tamil', 'nemo_id': 'ta', 'indic_code': 'tam_Taml'},
        'gu': {'name': 'Gujarati', 'nemo_id': 'gu', 'indic_code': 'guj_Gujr'},
        'kn': {'name': 'Kannada', 'nemo_id': 'kn', 'indic_code': 'kan_Knda'},
        'ml': {'name': 'Malayalam', 'nemo_id': 'ml', 'indic_code': 'mal_Mlym'},
        'pa': {'name': 'Punjabi', 'nemo_id': 'pa', 'indic_code': 'pan_Guru'},
        'or': {'name': 'Odia', 'nemo_id': 'or', 'indic_code': 'ory_Orya'}
    }
    
    def __init__(self, 
                 conformer_model_path: Optional[str] = None,
                 use_sarvam: bool = True,
                 device: Optional[str] = None):
        """
        Initialize the voice transcriber
        
        Args:
            conformer_model_path: Path to the Conformer .nemo model file
            use_sarvam: Whether to use SarvamAI as primary transcription service
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conformer_model_path = conformer_model_path
        self.use_sarvam = use_sarvam and HAS_SARVAM
        
        # Initialize models
        self.conformer_model = None
        self.indic_model = None
        self.indic_tokenizer = None
        self.indic_processor = None
        self.sarvam_client = None
        
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models if dependencies are available
        if HAS_NEMO:
            self._initialize_models()
        else:
            self.logger.warning("NeMo dependencies not available. Only SarvamAI transcription will work.")
    
    def _initialize_models(self):
        """Initialize AI4Bharat and IndicTrans models"""
        try:
            # Initialize IndicTrans2 for translation
            self._load_indic_trans()
            
            # Initialize Conformer model if path provided
            if self.conformer_model_path and os.path.exists(self.conformer_model_path):
                self._load_conformer_model()
            
            # Initialize SarvamAI client
            if self.use_sarvam:
                self._initialize_sarvam()
                
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            raise VoiceTranscriptionError(f"Failed to initialize models: {e}")
    
    def _load_indic_trans(self):
        """Load IndicTrans2 model for translation"""
        try:
            self.logger.info("Loading IndicTrans2 model...")
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
            
            self.indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                # attn_implementation="flash_attention_2"  # Comment out if causes issues
            ).to(self.device)
            
            self.indic_processor = IndicProcessor(inference=True)
            self.logger.info("IndicTrans2 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load IndicTrans2: {e}")
            raise
    
    def _load_conformer_model(self):
        """Load AI4Bharat Conformer model"""
        try:
            self.logger.info(f"Loading Conformer model from {self.conformer_model_path}")
            self.conformer_model = nemo_asr.models.EncDecCTCModel.restore_from(
                restore_path=self.conformer_model_path
            )
            self.conformer_model.eval()
            self.conformer_model = self.conformer_model.to(self.device)
            self.logger.info("Conformer model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Conformer model: {e}")
            raise
    
    def _initialize_sarvam(self):
        """Initialize SarvamAI client"""
        try:
            api_key = os.getenv('SARVAM_API_KEY') or os.getenv('sarvam_api')
            if not api_key:
                self.logger.warning("SarvamAI API key not found in environment variables")
                self.use_sarvam = False
                return
            
            self.sarvam_client = SarvamAI(api_subscription_key=api_key)
            self.logger.info("SarvamAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SarvamAI: {e}")
            self.use_sarvam = False
    
    def transcribe_audio(self, 
                        audio_path: str, 
                        language: str = 'mr',
                        translate_to_english: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio file and optionally translate to English
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'mr', 'hi', 'bn')
            translate_to_english: Whether to translate the transcription to English
        
        Returns:
            Dictionary containing transcription results
        """
        result = {
            'success': False,
            'transcription': '',
            'translation': '',
            'language': language,
            'method': 'unknown',
            'error': None
        }
        
        try:
            # Validate inputs
            if not os.path.exists(audio_path):
                raise VoiceTranscriptionError(f"Audio file not found: {audio_path}")
            
            if language not in self.SUPPORTED_LANGUAGES:
                raise VoiceTranscriptionError(f"Unsupported language: {language}")
            
            # Try SarvamAI first if available
            if self.use_sarvam and self.sarvam_client:
                try:
                    transcription = self._transcribe_with_sarvam(audio_path)
                    result.update({
                        'success': True,
                        'transcription': transcription,
                        'translation': transcription,  # SarvamAI already provides English
                        'method': 'sarvam'
                    })
                    self.logger.info("Transcription successful with SarvamAI")
                    return result
                except Exception as e:
                    self.logger.warning(f"SarvamAI transcription failed: {e}")
            
            # Fallback to Conformer + IndicTrans
            if self.conformer_model and self.indic_model:
                try:
                    transcription = self._transcribe_with_conformer(audio_path, language)
                    result['transcription'] = transcription
                    result['method'] = 'conformer'
                    
                    if translate_to_english and transcription:
                        translation = self._translate_with_indic_trans(
                            transcription, 
                            language
                        )
                        result['translation'] = translation
                    else:
                        result['translation'] = transcription
                    
                    result['success'] = True
                    self.logger.info("Transcription successful with Conformer+IndicTrans")
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Conformer transcription failed: {e}")
                    raise
            
            # If no models available
            raise VoiceTranscriptionError("No transcription models available")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Transcription failed: {e}")
            return result
    
    def _transcribe_with_sarvam(self, audio_path: str) -> str:
        """Transcribe using SarvamAI"""
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.sarvam_client.speech_to_text.translate(
                    file=audio_file,
                    model="saaras:v2.5"
                )
            
            # Extract transcript from response
            pattern = r"transcript='(.*?)'\s+language_code"
            match = re.search(pattern, str(response), re.DOTALL)
            
            if match:
                return match.group(1).strip()
            else:
                # Try alternative parsing
                response_str = str(response)
                if 'transcript=' in response_str:
                    start = response_str.find("transcript='") + 12
                    end = response_str.find("'", start)
                    return response_str[start:end].strip()
                
                raise VoiceTranscriptionError("Could not parse SarvamAI response")
                
        except Exception as e:
            raise VoiceTranscriptionError(f"SarvamAI transcription error: {e}")
    
    def _transcribe_with_conformer(self, audio_path: str, language: str) -> str:
        """Transcribe using AI4Bharat Conformer model"""
        try:
            self.conformer_model.cur_decoder = "ctc"
            lang_id = self.SUPPORTED_LANGUAGES[language]['nemo_id']
            
            results = self.conformer_model.transcribe(
                [audio_path], 
                batch_size=1,
                logprobs=False, 
                language_id=lang_id
            )
            
            if results and len(results) > 0:
                return results[0][0].strip()
            else:
                raise VoiceTranscriptionError("Empty transcription result")
                
        except Exception as e:
            raise VoiceTranscriptionError(f"Conformer transcription error: {e}")
    
    def _translate_with_indic_trans(self, text: str, source_language: str) -> str:
        """Translate text using IndicTrans2"""
        try:
            src_lang = self.SUPPORTED_LANGUAGES[source_language]['indic_code']
            tgt_lang = "eng_Latn"
            
            # Preprocess
            batch = self.indic_processor.preprocess_batch(
                [text], 
                src_lang=src_lang, 
                tgt_lang=tgt_lang
            )
            
            # Tokenize
            inputs = self.indic_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.indic_model.generate(
                    **inputs,
                    use_cache=False,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            # Decode
            generated_tokens = self.indic_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Postprocess
            translations = self.indic_processor.postprocess_batch(
                generated_tokens, 
                lang=tgt_lang
            )
            
            return translations[0].strip() if translations else ""
            
        except Exception as e:
            raise VoiceTranscriptionError(f"Translation error: {e}")
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get list of supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def is_model_ready(self) -> Dict[str, bool]:
        """Check which models are ready"""
        return {
            'sarvam': self.use_sarvam and self.sarvam_client is not None,
            'conformer': self.conformer_model is not None,
            'indic_trans': self.indic_model is not None,
            'has_nemo': HAS_NEMO
        }


def create_transcriber(conformer_model_path: Optional[str] = None, 
                      use_sarvam: bool = True) -> VoiceTranscriber:
    """
    Factory function to create a VoiceTranscriber instance
    
    Args:
        conformer_model_path: Path to the Conformer .nemo model file
        use_sarvam: Whether to use SarvamAI as primary transcription service
    
    Returns:
        VoiceTranscriber instance
    """
    return VoiceTranscriber(
        conformer_model_path=conformer_model_path,
        use_sarvam=use_sarvam
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the transcriber
    transcriber = create_transcriber()
    
    print("Voice Transcriber Status:")
    status = transcriber.is_model_ready()
    for model, ready in status.items():
        print(f"  {model}: {'✓' if ready else '✗'}")
    
    print("\nSupported Languages:")
    for code, info in transcriber.get_supported_languages().items():
        print(f"  {code}: {info['name']}")
