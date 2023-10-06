import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger

try:
    from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
except:
    logger.warning('Exllama module failed to load. Will attempt to load from repositories.')
    try:
        from modules.relative_imports import RelativeImport

        with RelativeImport("repositories/exllama"):
            from model import ExLlama, ExLlamaCache, ExLlamaConfig
    except:
        logger.error("Could not find repositories/exllama/. Make sure that exllama is cloned inside repositories/ and is up to date.")
        raise

def not_emoji_token(tokenId):
    # Byte level tokens are [3, 258] (we exclude 13 because it's linebreak)
    # High tokens (after 30245) are weird characters, including some emojis
    # Note that we treat high tokens similar as byte level tokens even though one high token maps to one emoji, whereas 4 byte tokens typically maps to one emoji (this is ugly but what do u do)
    return tokenId <= 2 or tokenId == 13 or (tokenId >= 259 and tokenId < 30245)

def get_emoji_penalty(seq):
    if len(seq) < 20:
        # Just a safety check, all sequences should in practice be larger than this
        return 1
    if not_emoji_token(seq[-1]) and not_emoji_token(seq[-2]):
        # Last 2 tokens are not emoji tokens, so we are not inside emoji repetition sequence
        return 1

    # Count consecutive byte tokens at the end (allowing last token to be non byte token, but stopping count when encountering the next non byte token)
    byte_token_count = 0 if not_emoji_token(seq[-1]) else 1
    i = -2
    while (i > -20):
        if not_emoji_token(seq[i]):
            break
        byte_token_count += 1
        i -= 1

    if byte_token_count < 4:
        # Most emojis need 4 byte tokens to complete, so allow the current emoji to be completed
        return 1
    if not_emoji_token(seq[-1]):
        # The last token ended a sequence of byte tokens, apply heavy penalty to prevent patterns like emoji-space-emoji-space...
        return 0.01

    # In the remaining cases last token is byte level token
    if byte_token_count == 4:
        # We probably completed an emoji, apply penalty to reduce probability of starting another emoji
        return 0.98
    if byte_token_count < 8:
        # We are probably in the middle of constructing a second consecutive emoji, allow it to complete
        return 1
    if byte_token_count == 8:
        # We probably completed second consecutive emoji, apply penalty to reduce probability of starting another emoji
        return 0.95
    if byte_token_count < 12:
        # We MIGHT be in the middle of constructing third consecutive emoji, but not all emojis need exactly 4 byte tokens, so maybe not
        return 0.97
    
    # If we have a byte sequence this long, we don't really care about completing emojis, just stop already.
    return 0.01

class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.generation_config = GenerationConfig()
        self.lora = None

        self.ex_cache = ExLlamaCache(self.ex_model)
        self.past_seq = None

        if shared.args.cfg_cache:
            self.ex_cache_negative = ExLlamaCache(self.ex_model)
            self.past_seq_negative = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not shared.args.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with ExLlama_HF.")
                return

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq_tensor[:min_length]))
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    ex_cache.current_seq_len = longest_prefix
                    if len(seq_tensor) - longest_prefix > 1:
                        self.ex_model.forward(seq_tensor[longest_prefix:-1].view(1, -1), ex_cache, preprocess_only=True, lora=self.lora)

            if reset:
                ex_cache.current_seq_len = 0
                if len(seq_tensor) > 1:
                    self.ex_model.forward(seq_tensor[:-1].view(1, -1), ex_cache, preprocess_only=True, lora=self.lora)

            logits = self.ex_model.forward(seq_tensor[-1:].view(1, -1), ex_cache, lora=self.lora).to(input_ids.device)
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(seq_tensor.view(1, -1), ex_cache, last_id_only=False, lora=self.lora)

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Custom sampler to reduce repetitive emoji runoffs
        emoji_penalty = get_emoji_penalty(seq)
        if emoji_penalty != 1:
            def apply_penalty(logits_slice):
                positive_mask = logits_slice > 0
                negative_mask = logits_slice < 0
                logits_slice[positive_mask] *= emoji_penalty
                logits_slice[negative_mask] /= emoji_penalty

            apply_penalty(logits[0, 0, 3:13])
            apply_penalty(logits[0, 0, 14:259])
            apply_penalty(logits[0, 0, 30245:32000])

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        pretrained_model_name_or_path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        config = ExLlamaConfig(pretrained_model_name_or_path / 'config.json')

        # from 'oobabooga/text-generation-webui/modules/exllama.py'
        weight_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert weight_path is not None, f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        config.max_seq_len = shared.args.max_seq_len
        config.compress_pos_emb = shared.args.compress_pos_emb
        if shared.args.gpu_split:
            config.set_auto_map(shared.args.gpu_split)
            config.gpu_peer_fix = True

        if shared.args.alpha_value > 1 and shared.args.rope_freq_base == 0:
            config.alpha_value = shared.args.alpha_value
            config.calculate_rotary_embedding_base()
        elif shared.args.rope_freq_base > 0:
            config.rotary_embedding_base = shared.args.rope_freq_base

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        # config.fused_attn = False
        # config.fused_mlp_thd = 0

        return ExllamaHF(config)
