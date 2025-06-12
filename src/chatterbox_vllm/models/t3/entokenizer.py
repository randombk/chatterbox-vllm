import logging
import os
from typing import List, Optional, Union

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer(PreTrainedTokenizer):
    """
    A VLLM-compatible tokenizer that wraps the original Tokenizer implementation.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
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
        vocab_file = os.path.join(os.path.dirname(__file__), "tokenizer.json")
        return cls(vocab_file=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = text.replace(' ', SPACE)
        return self.tokenizer.encode(text).tokens

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

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save the tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the tokenizer to
            **kwargs: Additional arguments to pass to the tokenizer
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        self.tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

    def text_to_tokens(self, text: str):
        """Legacy method for backward compatibility"""
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str, verbose=False, return_tensors: Optional[str] = None, add_special_tokens: bool = True):
        """Legacy method for backward compatibility"""
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        if return_tensors == "pt":
            return torch.IntTensor(ids).unsqueeze(0)
        return ids

    def decode(self, seq):
        """Legacy method for backward compatibility"""
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt

    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())