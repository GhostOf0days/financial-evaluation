# Import all necessary models to register them
try:
    from src.api_models import LocalCompletionsAPI, LocalChatCompletion, OpenAICompletionsAPI, OpenAIChatCompletion
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
        from src.api_models import LocalCompletionsAPI, LocalChatCompletion, OpenAICompletionsAPI, OpenAIChatCompletion
    except ImportError:
        print("Warning: Could not import custom API models from src.api_models")

from . import gpt2
from . import gpt3
try:
    from . import anthropic_llms
except ImportError:
    from . import anthropic
    anthropic_llms = anthropic
from . import huggingface
from . import textsynth
from . import dummy

try:
    from . import huggingface
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

# Dictionary for function registry
MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    # Comment out the problematic line
    # "hf-mlm": huggingface.AutoMLM,
    "hf-prefix-lm": huggingface.AutoPrefixLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "anthropic": anthropic_llms.AnthropicLM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}

try:
    # Check if AutoMLM exists and use it - for backward compatibility
    if hasattr(huggingface, "AutoMLM"):
        MODEL_REGISTRY["hf-mlm"] = huggingface.AutoMLM
        
    if hasattr(huggingface, "AutoLlamaCausalLM"):
        MODEL_REGISTRY["hf-causal-llama"] = huggingface.AutoLlamaCausalLM
    else:
        MODEL_REGISTRY["hf-causal-llama"] = huggingface.AutoCausalLM
        
    if hasattr(huggingface, "VLLM"):
        MODEL_REGISTRY["hf-causal-vllm"] = huggingface.VLLM
        
    if hasattr(huggingface, "AutoGLM"):
        MODEL_REGISTRY["hf-chatglm"] = huggingface.AutoGLM
except Exception as e:
    print(f"Warning: Could not restore all original models: {e}")

try:
    MODEL_REGISTRY.update({
        "local-completions": LocalCompletionsAPI,
        "local-chat-completions": LocalChatCompletion,
        "openai-completions": OpenAICompletionsAPI,
        "openai-chat-completions": OpenAIChatCompletion
    })
except NameError:
    print("Warning: Custom API models were not imported successfully, so they cannot be registered")

def get_model(model_name):
    return MODEL_REGISTRY[model_name]
