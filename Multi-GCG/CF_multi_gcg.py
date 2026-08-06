# Cross-Family Multi-Model GCG

import copy
import gc
import logging
import time
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg_cross_family")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def empty_cache():
    """Clear memory cache for CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True  # Only for gradient computation (batch=1)
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    # Gradient aggregation strategy
    # - "alternate": alternate between models each step
    # - "sum": sum gradients from all models (properly implemented now)
    # - "primary_only": only use primary model's gradient
    grad_aggregation: str = "alternate"
    primary_model_idx: int = 1  # Which model's tokenizer to use for sampling
    # NPU optimizations
    log_timing: bool = False


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBuffer:
    """Store candidate strings (not token IDs) to support cross-family models."""
    
    def __init__(self, size: int):
        self.buffer = []
        self.size = size

    def add(self, loss: float, optim_str: str) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_str)]
            return
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_str))
        else:
            self.buffer[-1] = (loss, optim_str)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_string(self) -> str:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self):
        message = "buffer:"
        for loss, optim_str in self.buffer:
            display_str = optim_str.replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\nloss: {loss:.4f} | string: {display_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
):
    """Sample new token IDs based on gradient information."""
    ids = ids.to(grad.device)
    n_optim_tokens = ids.shape[0]
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad = grad.clone()
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device),
        dim=1
    )[..., :n_replace]

    pick = torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    sampled_ids_val = torch.gather(topk_ids[sampled_ids_pos], 2, pick).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filter to keep only sequences that encode/decode consistently."""
    ids_cpu = ids.detach().cpu()
    ids_decoded = tokenizer.batch_decode(ids_cpu)

    filtered_ids = []
    for i in range(len(ids_decoded)):
        ids_encoded = tokenizer(
            ids_decoded[i],
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]
        if torch.equal(ids_cpu[i], ids_encoded):
            filtered_ids.append(ids_cpu[i])

    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    return torch.stack(filtered_ids, dim=0)


class ModelWrapper:
    """Wrapper for a model with its own tokenizer."""
    
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        name: str = "model"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = model.device
        
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        
        self.embedding_layer = model.get_input_embeddings()
        
        # These will be set during run()
        self.before_embeds_list = None
        self.after_embeds_list = None
        self.target_embeds = None
        self.target_ids = None
        self.prefix_cache_list = None
    
    def tokenize(self, text: str, add_special_tokens: bool = False) -> Tensor:
        """Tokenize text using this model's tokenizer."""
        return self.tokenizer(
            [text], 
            add_special_tokens=add_special_tokens, 
            return_tensors="pt"
        )["input_ids"].to(self.device, torch.int64)
    
    def batch_tokenize(
        self, 
        texts: List[str], 
        add_special_tokens: bool = False,
        padding: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Batch tokenize multiple texts. Returns (input_ids, attention_mask)."""
        encoded = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors="pt",
            truncation=False,
        )
        return (
            encoded["input_ids"].to(self.device, torch.int64),
            encoded["attention_mask"].to(self.device, torch.int64)
        )
    
    def embed(self, ids: Tensor) -> Tensor:
        """Convert token IDs to embeddings."""
        return self.embedding_layer(ids.to(self.device))
    
    def apply_chat_template(self, messages: List[dict]) -> str:
        """Apply chat template for this model."""
        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "", 1)
        return template
    
    def prepare_embeddings(
        self, 
        before_str_list: List[str], 
        after_str_list: List[str], 
        target: str,
        use_prefix_cache: bool = True  # Ignored - kept for API compatibility
    ):
        """Prepare all embeddings for this model."""
        self.target_ids = self.tokenize(target)
        self.target_embeds = self.embed(self.target_ids)
        
        self.before_embeds_list = [
            self.embed(self.tokenize(s)) for s in before_str_list
        ]
        self.after_embeds_list = [
            self.embed(self.tokenize(s)) for s in after_str_list
        ]
        
        # NOTE: prefix_cache is no longer used due to compatibility issues
        # with different transformers versions (tuple vs DynamicCache).
        # All forward passes now use full input concatenation.
        self.prefix_cache_list = None
    
    def clear_cache(self):
        """Clear internal caches to free memory."""
        empty_cache()  # Use the unified function


class CrossFamilyGCG:
    """
    GCG optimizer that supports models from different families with different tokenizers.
    """
    
    def __init__(
        self,
        models: List[ModelWrapper],
        config: GCGConfig,
    ):
        self.models = models
        self.config = config
        self.num_models = len(models)
        
        # Use primary model's tokenizer for sampling
        self.primary_idx = config.primary_model_idx - 1
        self.primary_model = models[self.primary_idx]
        self.primary_tokenizer = self.primary_model.tokenizer
        
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(
            self.primary_tokenizer, device=self.primary_model.device
        )
        
        self.stop_flag = False
        
        for i, wrapper in enumerate(models):
            m = wrapper.model
            logger.info(f"Model {wrapper.name}: dtype={m.dtype}, device={m.device}")
            if m.dtype in (torch.float32, torch.float64):
                logger.warning(
                    f"Model {wrapper.name} is in {m.dtype}. "
                    f"Use a lower precision data type for faster optimization."
                )
    
    def _prepare_messages_for_model(
        self, 
        wrapper: ModelWrapper, 
        multi_messages: List[Union[str, List[dict]]]
    ) -> Tuple[List[str], List[str]]:
        """Prepare before/after strings for a specific model using its tokenizer."""
        before_str_list, after_str_list = [], []
        
        for messages in multi_messages:
            messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else copy.deepcopy(messages)
            if not any("{optim_str}" in d["content"] for d in messages):
                messages[-1]["content"] += "{optim_str}"
            
            template = wrapper.apply_chat_template(messages)
            
            try:
                before_str, after_str = template.split("{optim_str}")
                before_str_list.append(before_str)
                after_str_list.append(after_str)
            except ValueError:
                raise ValueError(f"Each message must contain '{{optim_str}}' placeholder exactly once for {wrapper.name}")
        
        return before_str_list, after_str_list
    
    def run(
        self,
        multi_messages: List[Union[str, List[dict]]],
        target: str,
    ) -> GCGResult:
        config = self.config
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        target = (" " + target) if config.add_space_before_target else target
        
        # Prepare embeddings for each model
        logger.info("Preparing embeddings for all models...")
        for wrapper in self.models:
            before_str_list, after_str_list = self._prepare_messages_for_model(wrapper, multi_messages)
            wrapper.prepare_embeddings(
                before_str_list, after_str_list, target, 
                use_prefix_cache=config.use_prefix_cache
            )
        logger.info("Embeddings prepared.")
        
        # Initialize buffer with strings
        buffer = self.init_buffer()
        current_optim_str = buffer.get_best_string()
        
        losses, optim_strings = [], []
        timing_stats = {"grad": 0, "sample": 0, "loss": 0, "total": 0}
        
        for step in tqdm(range(config.num_steps)):
            step_start = time.time()
            
            # Convert current string to primary tokenizer's IDs
            optim_ids = self.primary_tokenizer(
                current_optim_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.primary_model.device)
            
            # =============================================
            # Compute gradient based on aggregation strategy
            # FIX #1: Properly implement all strategies
            # =============================================
            t0 = time.time()
            
            if config.grad_aggregation == "alternate":
                # Alternate between models
                model_idx = step % self.num_models
                grad = self.compute_token_gradient(optim_ids, model_idx)
                
            elif config.grad_aggregation == "sum":
                # FIX: Actually sum gradients from all models
                grad = None
                for model_idx in range(self.num_models):
                    model_grad = self.compute_token_gradient(optim_ids, model_idx)
                    if grad is None:
                        grad = model_grad
                    else:
                        grad = grad + model_grad
                grad = grad / self.num_models
                
            else:  # "primary_only" or default
                grad = self.compute_token_gradient(optim_ids, self.primary_idx)
            
            timing_stats["grad"] += time.time() - t0
            
            with torch.no_grad():
                # Sample new candidates using primary tokenizer
                t0 = time.time()
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    self.not_allowed_ids,
                ).cpu()
                
                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, self.primary_tokenizer)
                
                sampled_strings = self.primary_tokenizer.batch_decode(sampled_ids)
                timing_stats["sample"] += time.time() - t0
                
                # Compute aggregated loss
                t0 = time.time()
                batch_size = config.batch_size if config.batch_size is not None else len(sampled_strings)
                loss = self._compute_candidates_loss_by_string(sampled_strings, batch_size)
                timing_stats["loss"] += time.time() - t0
                
                best_i = loss.argmin().item()
                current_loss = loss[best_i].item()
                current_optim_str = sampled_strings[best_i]
                
                losses.append(current_loss)
                
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, current_optim_str)
            
            current_optim_str = buffer.get_best_string()
            optim_strings.append(current_optim_str)
            
            timing_stats["total"] += time.time() - step_start
            
            buffer.log_buffer()
            
            if config.log_timing and (step + 1) % 10 == 0:
                avg_time = timing_stats["total"] / (step + 1)
                logger.info(
                    f"Avg time/step: {avg_time:.2f}s "
                    f"(grad: {timing_stats['grad']/(step+1):.2f}s, "
                    f"sample: {timing_stats['sample']/(step+1):.2f}s, "
                    f"loss: {timing_stats['loss']/(step+1):.2f}s)"
                )
            
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break
        
        min_loss_index = losses.index(min(losses))
        return GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )
    
    def init_buffer(self) -> AttackBuffer:
        config = self.config
        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")
        buffer = AttackBuffer(config.buffer_size)
        
        if isinstance(config.optim_str_init, str):
            init_strings = [config.optim_str_init]
            
            if config.buffer_size > 1:
                init_char_ids = self.primary_tokenizer(
                    INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].squeeze(0).cpu()
                
                init_optim_ids = self.primary_tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].cpu()
                
                for _ in range(config.buffer_size - 1):
                    rand_ids = init_char_ids[
                        torch.randint(0, init_char_ids.shape[0], (init_optim_ids.shape[1],))
                    ].unsqueeze(0)
                    init_strings.append(self.primary_tokenizer.batch_decode(rand_ids)[0])
        else:
            init_strings = config.optim_str_init
        
        for s in init_strings:
            loss = self._compute_single_string_loss(s)
            buffer.add(loss, s)
        
        buffer.log_buffer()
        logger.info("Initialized attack buffer.")
        return buffer
    
    def compute_token_gradient(self, optim_ids: Tensor, model_idx: int) -> Tensor:
        """Compute gradient for token replacement using the specified model."""
        wrapper = self.models[model_idx]
        
        # If using primary model, compute directly
        if model_idx == self.primary_idx:
            return self._compute_gradient_direct(wrapper, optim_ids)
        
        # For non-primary models, check token alignment
        optim_str = self.primary_tokenizer.batch_decode(optim_ids)[0]
        target_optim_ids = wrapper.tokenize(optim_str).squeeze(0)
        
        # FIX #4: Better handling of token count mismatch
        # Log the mismatch for debugging
        if target_optim_ids.shape[0] != optim_ids.shape[1]:
            logger.debug(
                f"Token count mismatch for model {wrapper.name}: "
                f"primary={optim_ids.shape[1]}, target={target_optim_ids.shape[0]}. "
                f"Using primary gradient instead."
            )
            return self._compute_gradient_direct(self.primary_model, optim_ids)
        
        # Compute gradient in target model's space
        target_grad = self._compute_gradient_direct(wrapper, target_optim_ids.unsqueeze(0))
        
        # Map to primary space
        return self._map_gradient_to_primary_space(
            target_grad.squeeze(0), target_optim_ids, optim_ids.squeeze(0), wrapper
        ).unsqueeze(0)
    
    def _compute_gradient_direct(self, wrapper: ModelWrapper, optim_ids: Tensor) -> Tensor:
        """
        Compute gradient directly using a single model.
        
        NOTE: We don't use prefix_cache here because:
        1. New transformers versions use Cache objects, not tuples
        2. Gradient computation is batch_size=1, so overhead is acceptable
        3. Avoids compatibility issues across transformers versions
        """
        model = wrapper.model
        embedding_layer = wrapper.embedding_layer
        
        optim_ids_device = optim_ids.to(model.device)
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids_device.squeeze(0), num_classes=embedding_layer.num_embeddings
        ).to(model.dtype)
        optim_ids_onehot.requires_grad_(True)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight
        optim_embeds = optim_embeds.unsqueeze(0)
        
        total_loss = 0
        num_prompts = len(wrapper.before_embeds_list)
        
        for i in range(num_prompts):
            # Always use full sequence without prefix_cache for gradient computation
            # This avoids Cache format compatibility issues
            input_embeds = torch.cat([
                wrapper.before_embeds_list[i], 
                optim_embeds, 
                wrapper.after_embeds_list[i], 
                wrapper.target_embeds
            ], dim=1)
            output = model(inputs_embeds=input_embeds, use_cache=False)
            
            logits = output.logits
            shift = input_embeds.shape[1] - wrapper.target_ids.shape[1]
            shift_logits = logits[..., shift - 1: -1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                wrapper.target_ids.view(-1)
            )
            total_loss += loss
        
        grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]
        return grad.unsqueeze(0)
    
    def _map_gradient_to_primary_space(
        self, 
        target_grad: Tensor, 
        target_ids: Tensor,
        primary_ids: Tensor,
        wrapper: ModelWrapper
    ) -> Tensor:
        """Map gradient from target model's token space to primary model's token space."""
        primary_vocab_size = len(self.primary_tokenizer)
        primary_grad = torch.zeros(
            primary_ids.shape[0], primary_vocab_size,
            device=self.primary_model.device, dtype=target_grad.dtype
        )
        
        top_k = min(128, target_grad.shape[-1])
        
        for pos in range(primary_ids.shape[0]):
            top_target_indices = (-target_grad[pos]).topk(top_k).indices
            
            for target_tok_id in top_target_indices:
                target_tok_str = wrapper.tokenizer.decode([target_tok_id.item()])
                primary_tok_ids = self.primary_tokenizer.encode(
                    target_tok_str, add_special_tokens=False
                )
                if len(primary_tok_ids) == 1:
                    primary_grad[pos, primary_tok_ids[0]] = target_grad[pos, target_tok_id]
        
        return primary_grad
    
    def _compute_candidates_loss_by_string(
        self, 
        candidate_strings: List[str], 
        batch_size: int
    ) -> Tensor:
        """Compute aggregated loss for candidate strings across all models."""
        num_candidates = len(candidate_strings)
        
        model_losses = []
        for wrapper in self.models:
            model_loss = self._compute_single_model_loss_no_cache(
                wrapper, candidate_strings, batch_size
            )
            model_losses.append(model_loss)
        
        total_loss = torch.zeros(num_candidates, device=self.primary_model.device)
        for loss in model_losses:
            total_loss += loss.to(self.primary_model.device)
        
        return total_loss / self.num_models
    
    def _compute_single_model_loss_no_cache(
        self,
        wrapper: ModelWrapper,
        candidate_strings: List[str],
        batch_size: int
    ) -> Tensor:
        """
        Compute loss for candidate strings on a single model.
        NO PREFIX_CACHE to avoid OOM.
        Properly handles attention_mask for padding.
        """
        num_candidates = len(candidate_strings)
        all_loss = torch.zeros(num_candidates, device=wrapper.device)
        num_prompts = len(wrapper.before_embeds_list)
        
        # Pre-tokenize all candidates
        all_ids, all_masks = wrapper.batch_tokenize(candidate_strings)
        
        for prompt_idx in range(num_prompts):
            before_embeds = wrapper.before_embeds_list[prompt_idx]
            after_embeds = wrapper.after_embeds_list[prompt_idx]
            
            before_len = before_embeds.shape[1]
            after_len = after_embeds.shape[1]
            target_len = wrapper.target_embeds.shape[1]
            
            for j in range(0, num_candidates, batch_size):
                with torch.no_grad():
                    current_batch_size = min(batch_size, num_candidates - j)
                    
                    batch_ids = all_ids[j:j + current_batch_size]
                    batch_masks = all_masks[j:j + current_batch_size]
                    optim_embeds_batch = wrapper.embedding_layer(batch_ids)
                    
                    input_embeds_batch = torch.cat([
                        before_embeds.expand(current_batch_size, -1, -1),
                        optim_embeds_batch,
                        after_embeds.expand(current_batch_size, -1, -1),
                        wrapper.target_embeds.expand(current_batch_size, -1, -1),
                    ], dim=1)
                    
                    # FIX #6: Proper attention_mask
                    attention_mask = torch.cat([
                        torch.ones(current_batch_size, before_len, device=wrapper.device, dtype=batch_masks.dtype),
                        batch_masks,
                        torch.ones(current_batch_size, after_len, device=wrapper.device, dtype=batch_masks.dtype),
                        torch.ones(current_batch_size, target_len, device=wrapper.device, dtype=batch_masks.dtype),
                    ], dim=1)
                    
                    outputs = wrapper.model(
                        inputs_embeds=input_embeds_batch,
                        attention_mask=attention_mask,
                        use_cache=False  # FIX #2: prevent cache leak
                    )
                    
                    logits = outputs.logits
                    shift = input_embeds_batch.shape[1] - wrapper.target_ids.shape[1]
                    shift_logits = logits[..., shift - 1: -1, :].contiguous()
                    
                    shift_labels = wrapper.target_ids.to(shift_logits.device).expand(current_batch_size, -1)
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        reduction="none",
                    )
                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                    all_loss[j:j + current_batch_size] += loss
                    
                    if self.config.early_stop and not self.stop_flag:
                        if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                            self.stop_flag = True
                    
                    del outputs, logits, shift_logits, input_embeds_batch, optim_embeds_batch, attention_mask
        
        return all_loss / num_prompts
    
    def _compute_single_string_loss(self, optim_str: str) -> float:
        """Compute aggregated loss for a single string."""
        total_loss = 0.0
        
        for wrapper in self.models:
            model_loss = self._compute_single_model_loss_no_cache(
                wrapper, [optim_str], batch_size=1
            )
            total_loss += model_loss[0].item()
        
        return total_loss / self.num_models


def _is_chat_messages(x) -> bool:
    return (
        isinstance(x, list)
        and len(x) > 0
        and isinstance(x[0], dict)
        and ("role" in x[0])
        and ("content" in x[0])
    )


def run(
    models: List[Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, str]],
    messages: Union[str, List[dict], List[Union[str, List[dict]]]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Run cross-family GCG attack."""
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    model_wrappers = [
        ModelWrapper(m, tok, name) for m, tok, name in models
    ]
    
    gcg = CrossFamilyGCG(model_wrappers, config)
    
    if isinstance(messages, str) or _is_chat_messages(messages):
        messages = [messages]
    
    result = gcg.run(messages, target)
    return result


def run_two_models(
    model1: transformers.PreTrainedModel,
    model2: transformers.PreTrainedModel,
    tokenizer1: transformers.PreTrainedTokenizer,
    tokenizer2: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict], List[Union[str, List[dict]]]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Convenience function for running with exactly two models."""
    models = [
        (model1, tokenizer1, "model1"),
        (model2, tokenizer2, "model2"),
    ]
    return run(models, messages, target, config)
