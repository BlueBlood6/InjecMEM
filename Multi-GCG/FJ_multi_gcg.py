import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


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
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor: return self.buffer[0][1]
    def get_lowest_loss(self) -> float: return self.buffer[0][0]
    def get_highest_loss(self) -> float: return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0].replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(ids: Tensor, grad: Tensor, search_width: int, topk: int = 256, n_replace: int = 1, not_allowed_ids: Tensor = False):
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)
    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")
    topk_ids = (-grad).topk(topk, dim=1).indices
    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(topk_ids[sampled_ids_pos], 2, torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)).squeeze(2)
    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []
    for i in range(len(ids_decoded)):
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])
    if not filtered_ids:
        raise RuntimeError("No token sequences are the same after decoding and re-encoding. Consider setting `filter_ids=False` or trying a different `optim_str_init`")
    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model1: transformers.PreTrainedModel,
        model2: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer1 = self.model1.get_input_embeddings()
        self.embedding_layer2 = self.model2.get_input_embeddings()
        
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model1.device)
        
        self.prefix_cache_list1 = None 
        self.prefix_cache_list2 = None

        self.stop_flag = False
        
        for model in [self.model1, self.model2]:
            if model.dtype in (torch.float32, torch.float64):
                logger.warning(f"Model {model.name_or_path} is in {model.dtype}. Use a lower precision data type for faster optimization.")
            if model.device == torch.device("cpu"):
                logger.warning(f"Model {model.name_or_path} is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def run(
        self,
        multi_messages: List[Union[str, List[dict]]],
        target: str,
    ) -> GCGResult:
        config = self.config
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        before_str_list, after_str_list = [], []
        for messages in multi_messages:
            messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else copy.deepcopy(messages)
            if not any(["{optim_str}" in d["content"] for d in messages]):
                messages[-1]["content"] += "{optim_str}"
            
            template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
                template = template.replace(self.tokenizer.bos_token, "")
            
            try:
                before_str, after_str = template.split("{optim_str}")
                before_str_list.append(before_str)
                after_str_list.append(after_str)
            except ValueError:
                raise ValueError("Each message must contain '{optim_str}' placeholder exactly once.")

        target = " " + target if config.add_space_before_target else target

        device1, device2 = self.model1.device, self.model2.device
        
        self.target_ids1 = self.tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device1, torch.int64)
        self.target_ids2 = self.tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device2, torch.int64)

        self.before_embeds_list1 = [self.embedding_layer1(self.tokenizer([s], padding=False, return_tensors="pt")["input_ids"].to(device1, torch.int64)) for s in before_str_list]
        self.after_embeds_list1 = [self.embedding_layer1(self.tokenizer([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device1, torch.int64)) for s in after_str_list]
        self.target_embeds1 = self.embedding_layer1(self.target_ids1)

        self.before_embeds_list2 = [self.embedding_layer2(self.tokenizer([s], padding=False, return_tensors="pt")["input_ids"].to(device2, torch.int64)) for s in before_str_list]
        self.after_embeds_list2 = [self.embedding_layer2(self.tokenizer([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device2, torch.int64)) for s in after_str_list]
        self.target_embeds2 = self.embedding_layer2(self.target_ids2)

        if config.use_prefix_cache:
            self.prefix_cache_list1, self.prefix_cache_list2 = [], []
            with torch.no_grad():
                for before_embeds in self.before_embeds_list1:
                    output = self.model1(inputs_embeds=before_embeds, use_cache=True)
                    self.prefix_cache_list1.append(output.past_key_values)
                for before_embeds in self.before_embeds_list2:
                    output = self.model2(inputs_embeds=before_embeds, use_cache=True)
                    self.prefix_cache_list2.append(output.past_key_values)

        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()
        losses, optim_strings = [], []

        for step in tqdm(range(config.num_steps)):
            model_idx_to_use = 1 if step % 2 == 0 else 2
            logger.debug(f"Step {step}: Using gradient from Model {model_idx_to_use}")

            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids, model_idx=model_idx_to_use)
            
            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0), optim_ids_onehot_grad.squeeze(0),
                    config.search_width, config.topk, config.n_replace, self.not_allowed_ids
                )
                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, self.tokenizer)
                
                batch_size = config.batch_size if config.batch_size is not None else sampled_ids.shape[0]
                loss = find_executable_batch_size(self._compute_candidates_loss, batch_size)(sampled_ids)
                
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            buffer.log_buffer(self.tokenizer)
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break
        
        min_loss_index = losses.index(min(losses))
        return GCGResult(
            best_loss=losses[min_loss_index], best_string=optim_strings[min_loss_index],
            losses=losses, strings=optim_strings
        )

    def init_buffer(self) -> AttackBuffer:
        config = self.config
        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")
        buffer = AttackBuffer(config.buffer_size)
        
        # This part assumes optim_str_init can be tokenized by the shared tokenizer
        device = self.model1.device # Use one device for init_buffer_ids
        if isinstance(config.optim_str_init, str):
            init_optim_ids = self.tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            if config.buffer_size > 1:
                init_buffer_ids = self.tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
        else: # list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = self.tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)
        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss, true_buffer_size)(init_buffer_ids)
        
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i].item(), init_buffer_ids[[i]])
        
        buffer.log_buffer(self.tokenizer)
        logger.info("Initialized attack buffer.")
        return buffer

    def compute_token_gradient(self, optim_ids: Tensor, model_idx: int) -> Tensor:
        
        total_loss = 0
        optim_ids_onehot, embedding_layer = None, None
        
        if model_idx == 1:
            model = self.model1
            embedding_layer = self.embedding_layer1
            optim_ids_device = optim_ids.to(model.device)
            optim_ids_onehot = torch.nn.functional.one_hot(optim_ids_device, num_classes=embedding_layer.num_embeddings).to(model.dtype)
            optim_embeds = optim_ids_onehot.requires_grad_() @ embedding_layer.weight
            
            before_embeds_list = self.before_embeds_list1
            after_embeds_list = self.after_embeds_list1
            target_embeds = self.target_embeds1
            target_ids = self.target_ids1
            prefix_cache_list = self.prefix_cache_list1

        elif model_idx == 2:
            model = self.model2
            embedding_layer = self.embedding_layer2
            optim_ids_device = optim_ids.to(model.device)
            optim_ids_onehot = torch.nn.functional.one_hot(optim_ids_device, num_classes=embedding_layer.num_embeddings).to(model.dtype)
            optim_embeds = optim_ids_onehot.requires_grad_() @ embedding_layer.weight

            before_embeds_list = self.before_embeds_list2
            after_embeds_list = self.after_embeds_list2
            target_embeds = self.target_embeds2
            target_ids = self.target_ids2
            prefix_cache_list = self.prefix_cache_list2
        else:
            raise ValueError("model_idx must be 1 or 2")

        num_prompts = len(before_embeds_list)
        for i in range(num_prompts):
            if prefix_cache_list:
                input_embeds = torch.cat([optim_embeds, after_embeds_list[i], target_embeds], dim=1)
                output = model(inputs_embeds=input_embeds, past_key_values=prefix_cache_list[i])
            else:
                input_embeds = torch.cat([before_embeds_list[i], optim_embeds, after_embeds_list[i], target_embeds], dim=1)
                output = model(inputs_embeds=input_embeds)
            
            logits = output.logits
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1 : -1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
            total_loss += loss

        grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]
        
        return grad

    def _compute_candidates_loss(self, search_batch_size: int, sampled_ids: Tensor) -> Tensor:
        
        # --- Loss from Model 1 ---
        loss1 = self._compute_single_model_loss(
            self.model1, self.embedding_layer1, sampled_ids, search_batch_size,
            self.before_embeds_list1, self.after_embeds_list1, self.target_embeds1,
            self.target_ids1, self.prefix_cache_list1
        )
        
        # --- Loss from Model 2 ---
        loss2 = self._compute_single_model_loss(
            self.model2, self.embedding_layer2, sampled_ids, search_batch_size,
            self.before_embeds_list2, self.after_embeds_list2, self.target_embeds2,
            self.target_ids2, self.prefix_cache_list2
        )
        
        # Combine losses, ensuring they are on the same device
        return (loss1 + loss2.to(loss1.device)) / 2.0

    def _compute_single_model_loss(
        self, model, embedding_layer, sampled_ids, search_batch_size,
        before_embeds_list, after_embeds_list, target_embeds, target_ids, prefix_cache_list
    ) -> Tensor:
        # This is a helper function refactored from your original _compute_candidates_loss_original
        all_loss_total = torch.zeros(sampled_ids.shape[0], device=model.device)
        num_prompts = len(before_embeds_list)
        
        sampled_ids = sampled_ids.to(model.device)

        for i in range(num_prompts):
            before_embeds = before_embeds_list[i]
            after_embeds = after_embeds_list[i]
            prefix_cache = prefix_cache_list[i] if self.config.use_prefix_cache else None
            
            prefix_cache_batch = None

            for j in range(0, sampled_ids.shape[0], search_batch_size):
                with torch.no_grad():
                    batch_ids = sampled_ids[j:j + search_batch_size]
                    current_batch_size = batch_ids.shape[0]
                    optim_embeds_batch = embedding_layer(batch_ids)

                    if prefix_cache:
                        input_embeds_batch = torch.cat([
                            optim_embeds_batch,
                            after_embeds.repeat(current_batch_size, 1, 1),
                            target_embeds.repeat(current_batch_size, 1, 1)
                        ], dim=1)
                        if not prefix_cache_batch or current_batch_size != search_batch_size:
                             prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in layer_cache] for layer_cache in prefix_cache]
                        outputs = model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                    else:
                        input_embeds_batch = torch.cat([
                            before_embeds.repeat(current_batch_size, 1, 1), optim_embeds_batch,
                            after_embeds.repeat(current_batch_size, 1, 1), target_embeds.repeat(current_batch_size, 1, 1),
                        ], dim=1)
                        outputs = model(inputs_embeds=input_embeds_batch)
                    
                    logits = outputs.logits
                    tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
                    shift_logits = logits[..., tmp-1:-1, :].contiguous()
                    shift_labels = target_ids.repeat(current_batch_size, 1)

                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                    all_loss_total[j:j + search_batch_size] += loss
                    
                    if self.config.early_stop and not self.stop_flag:
                        if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                            self.stop_flag = True
        
        return all_loss_total / num_prompts

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model1: transformers.PreTrainedModel,
    model2: transformers.PreTrainedModel, # MODIFIED
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict], List[Union[str, List[dict]]]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))

    # MODIFIED: Instantiate with two models
    gcg = GCG(model1, model2, tokenizer, config)

    is_multi_prompt = isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], (list, str, dict))
    if not is_multi_prompt:
        messages = [messages]

    result = gcg.run(messages, target)
    return result