import pytest
import torch


def _tiny(arch: str, layers: int = 6):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    if arch == "llama":
        m = LlamaForCausalLM(LlamaConfig(**kw))
    else:
        m = Qwen2ForCausalLM(Qwen2Config(**kw))
    return m.eval()


@pytest.fixture(params=["llama", "qwen2"])
def tiny_model(request):
    return _tiny(request.param)
