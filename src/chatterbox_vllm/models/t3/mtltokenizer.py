import logging
import os
import time
from typing import List, Optional, Union
from pathlib import Path
import json
from unicodedata import category, normalize

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
# _kakasi = None
_dicta = None
_russian_stresser = None

# Cache for prebatched dicta results
_hebrew_prebatch_cache = {}


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    global _hebrew_prebatch_cache
    
    # Check if this text was prebatched
    if text in _hebrew_prebatch_cache:
        return _hebrew_prebatch_cache[text]
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            # Path to the downloaded ONNX model (in workspace root)
            # Go up 5 levels from this file to workspace root: models/t3/mtltokenizer.py -> models -> chatterbox_vllm -> src -> workspace
            model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "dicta-1.0.int8.onnx"
            if model_path.exists():
                _dicta = Dicta(model_path=str(model_path))
                logger.info(f"Loaded Hebrew diacritization model from {model_path}")
            else:
                logger.warning(f"Hebrew diacritization model not found at {model_path} - skipping")
                return text
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def batch_add_hebrew_diacritics(texts: List[str]) -> List[str]:
    """Batch Hebrew diacritization for better performance."""
    global _dicta
    global _hebrew_prebatch_cache
    
    if not texts:
        return []
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "dicta-1.0.int8.onnx"
            if model_path.exists():
                _dicta = Dicta(model_path=str(model_path))
                logger.info(f"Loaded Hebrew diacritization model from {model_path}")
            else:
                logger.warning(f"Hebrew diacritization model not found at {model_path} - skipping")
                return texts
        
        # Use the underlying model.predict() which supports true batching
        results = _dicta.model.predict(texts, mark_matres_lectionis=None)
        
        # Cache the results for later use during tokenization
        for text, result in zip(texts, results):
            _hebrew_prebatch_cache[text] = result
        
        return results
            
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return texts
    except Exception as e:
        logger.warning(f"Batch Hebrew diacritization failed: {e}")
        return texts


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping()
        self._init_segmenter()
    
    def _load_cangjie_mapping(self):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def add_russian_stress(text: str) -> str:
    """Russian text normalization: adds stress marks to Russian text."""
    global _russian_stresser
    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser
            _russian_stresser = RussianTextStresser()
        return _russian_stresser.stress_text(text)
    except ImportError:
        logger.warning("russian_text_stresser not available - Russian stress labeling skipped")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text


class MTLTokenizer(PreTrainedTokenizer):
    """
    A VLLM-compatible tokenizer that wraps the original MTLTokenizer implementation.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file_path: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.cangjie_converter = ChineseCangjieConverter()
        self.check_vocabset_sot_eot()

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Instantiate a tokenizer from a pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Path to the tokenizer file or model name
            **kwargs: Additional arguments to pass to the tokenizer
        """
        # Load relative to the current file path
        vocab_file = os.path.join(os.path.dirname(__file__), "grapheme_mtl_merged_expanded_v1.json")
        return cls(vocab_file_path=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def preprocess_text(self, raw_text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text
    
    def prebatch_hebrew_texts(self, prompts: List[str], language_id: str = 'he') -> None:
        """
        Pre-batch Hebrew diacritization for all prompts before tokenization.
        This should be called before the tokenizer processes individual prompts.
        """
        if language_id != 'he':
            return
        
        # Extract the actual text from prompts (remove [START], <he>, [STOP])
        hebrew_texts = []
        for prompt in prompts:
            text = prompt
            # Remove [START]
            if text.startswith('[START]'):
                text = text[7:]
            # Remove [STOP]
            if text.endswith('[STOP]'):
                text = text[:-6]
            # Remove language token
            if text.startswith('<'):
                text = text.split('>')[1] if '>' in text else text
            
            # Preprocess (lowercase, normalize)
            text = self.preprocess_text(text, language_id)
            hebrew_texts.append(text)
        
        # Batch process all Hebrew texts
        if hebrew_texts:
            print(f"[DICTA-BATCH] Processing {len(hebrew_texts)} Hebrew texts")
            start = time.time()
            batch_add_hebrew_diacritics(hebrew_texts)
            elapsed = time.time() - start
            print(f"[DICTA-BATCH] Completed in {elapsed:.3f}s ({len(hebrew_texts)/elapsed:.1f} texts/sec)")

    def _tokenize(self, text: str, **kwargs) -> List[str]:        
        # Parse out language token if it exists
        # This is injected by the ChatterboxTTS.generate_with_conds method
        # It can be at the start or after [START]
        language_id = None
        prefix = ""
        suffix = ""
        
        # Check if text starts with [START]<lang>
        if text.startswith('[START]<'):
            prefix = '[START]'
            text = text[7:]  # Remove [START]
        
        # Check if text ends with [STOP]
        if text.endswith('[STOP]'):
            suffix = '[STOP]'
            text = text[:-6]  # Remove [STOP]
        
        # Now check for language token
        if text.startswith('<'):
            language_id = text.split('<')[1].split('>')[0]
            text = text.split('>')[1]
        
        text = self.preprocess_text(text, language_id)
        
        # Language-specific text processing
        if language_id == 'zh':
            text = self.cangjie_converter(text)
        elif language_id == 'ja':
            text = hiragana_normalize(text)
        elif language_id == 'he':
            text = add_hebrew_diacritics(text)
        elif language_id == 'ko':
            text = korean_normalize(text)
        elif language_id == 'ru':
            text = add_russian_stress(text)
        
        # Prepend language token again
        if language_id:
            text = f"[{language_id.lower()}]{text}"
        
        # Add back the [START] prefix and [STOP] suffix if they were there
        if prefix:
            text = prefix + text
        if suffix:
            text = text + suffix
        
        text = text.replace(' ', SPACE)
        return self.tokenizer.encode(text).tokens
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        **kwargs
    ):
        """Batch encoding with language-specific preprocessing."""
        # Process all texts through language-specific preprocessing
        processed_texts = []
        for text in batch_text_or_text_pairs:
            # Same logic as _tokenize but for batch processing
            language_id = None
            prefix = ""
            suffix = ""
            
            if text.startswith('[START]<'):
                prefix = '[START]'
                text = text[7:]
            
            if text.endswith('[STOP]'):
                suffix = '[STOP]'
                text = text[:-6]
            
            if text.startswith('<'):
                language_id = text.split('<')[1].split('>')[0]
                text = text.split('>')[1]
            
            text = self.preprocess_text(text, language_id)
            
            # Language-specific text processing
            if language_id == 'zh':
                text = self.cangjie_converter(text)
            elif language_id == 'ja':
                text = hiragana_normalize(text)
            elif language_id == 'he':
                text = add_hebrew_diacritics(text)
            elif language_id == 'ko':
                text = korean_normalize(text)
            elif language_id == 'ru':
                text = add_russian_stress(text)
            
            if language_id:
                text = f"[{language_id.lower()}]{text}"
            
            if prefix:
                text = prefix + text
            if suffix:
                text = text + suffix
            
            text = text.replace(' ', SPACE)
            processed_texts.append(text)
        
        # Batch tokenize all processed texts
        encodings = [self.tokenizer.encode(text) for text in processed_texts]
        
        # Convert to IDs
        input_ids = [enc.ids for enc in encodings]
        
        result = {'input_ids': input_ids}
        
        if return_attention_mask:
            result['attention_mask'] = [[1] * len(ids) for ids in input_ids]
        
        return result

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())