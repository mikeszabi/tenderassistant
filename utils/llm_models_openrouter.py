# OpenRouter integration for LLM models with performance and cost tracking
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Any, Dict, List, Optional, Iterator, ClassVar, Type, Tuple
from dotenv import load_dotenv
import time
import json
import os
import requests
from datetime import datetime
from pathlib import Path

# Env variables
load_dotenv()

# Create a directory for storing metrics if it doesn't exist
METRICS_DIR = Path(__file__).parent.parent / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

# Available models for benchmarking
AVAILABLE_MODELS = [
    # OpenAI models
    "gpt-4o",  # OpenAI reference model
    "gpt-4.1-mini",  # OpenAI mini model
    "gpt-4.1-nano",  # OpenAI nano model
    
    # OpenRouter models (using OpenAI API)
    "openai/gpt-4o",  # OpenRouter reference model
    "openai/gpt-4.1-mini",  # OpenRouter mini model

    # Anthropic models
    "anthropic/claude-3-opus",  # Anthropic Opus model
    "anthropic/claude-3-sonnet",  # Anthropic Sonnet model
    "anthropic/claude-3.7-sonnet",  # Anthropic 3.7 Sonnet model
    "anthropic/claude-3.7-haiku",  # Anthropic 3.7 Haiku model
    
    # Meta models
    "meta-llama/llama-3-70b-instruct",  # Meta Llama 3 70B model
    
    # Google models
    "google/gemini-2.0-flash-001",  # Google Gemini 2.0 Flash model
    
    # Mistral models
    "mistralai/mistral-large",  # Mistral large model
    
    # DeepSeek models
    "deepseek/deepseek-r1-0528", # DeepSeek R1 model
    "deepseek/deepseek-r1-0528:free",  # DeepSeek R1 free model
    "deepseek/deepseek-chat-v3-0324:free",  # DeepSeek free model

]

# Model pricing (approximate cost per 1K tokens in USD)
# Updated July 2025 based on OpenRouter pricing
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},

    # Openrouter models
    "openai/gpt-4o": {"input": 0.005, "output": 0.015},
    "openai/gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
    # "openai/gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    
    # Anthropic models
    "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
    "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3.7-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3.7-haiku": {"input": 0.00025, "output": 0.00125},
    
    # Meta models
    "meta-llama/llama-3-70b-instruct": {"input": 0.0009, "output": 0.0009},
    
    # Google models
    "google/gemini-2.0-flash-001": {"input": 0.0001, "output": 0.0002},
    
    # Mistral models
    "mistralai/mistral-large": {"input": 0.002, "output": 0.006},
    
    # DeepSeek models
    "deepseek/deepseek-r1-0528": {"input": 0.0005, "output": 0.00215},
    "deepseek/deepseek-r1-0528:free": {"input": 0.0000, "output": 0.000},
    "deepseek/deepseek-chat-v3-0324:free": {"input": 0.0000, "output": 0.000},
    
    # Meta models
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0000, "output": 0.000},

    # Default fallback pricing if model not found
    "default": {"input": 0.01, "output": 0.03}
}

# Dictionary to store model information
MODEL_INFO = {
    # OpenAI models for reference
    "gpt-4o": {"provider": "openai", "type": "chat"},
    "gpt-4.1-mini": {"provider": "openai", "type": "chat"},
    "gpt-4.1-nano": {"provider": "openai", "type": "chat"},
    
    # OpenRouter models
    "openai/gpt-4o": {"provider": "openrouter", "type": "chat"},
    "openai/gpt-4.1-mini": {"provider": "openrouter", "type": "chat"},
    # "openai/gpt-4.1-nano": {"provider": "openai", "type": "chat"},
   
    # Anthropic models
    "anthropic/claude-3-opus": {"provider": "openrouter", "type": "chat"},
    "anthropic/claude-3-sonnet": {"provider": "openrouter", "type": "chat"},
    "anthropic/claude-3.7-sonnet": {"provider": "openrouter", "type": "chat"},
    "anthropic/claude-3.7-haiku": {"provider": "openrouter", "type": "chat"},
    
    # Meta models
    "meta-llama/llama-3-70b-instruct": {"provider": "openrouter", "type": "chat"},
    
    # Google models
    "google/gemini-2.0-flash-001": {"provider": "openrouter", "type": "chat"},
    
    # Mistral models
    "mistralai/mistral-large": {"provider": "openrouter", "type": "chat"},
    
    # DeepSeek models
    "deepseek/deepseek-r1-0528": {"provider": "openrouter", "type": "chat"},
    "deepseek/deepseek-r1-0528:free": {"provider": "openrouter", "type": "chat"},
    "deepseek/deepseek-chat-v3-0324:free": {"provider": "openrouter", "type": "chat"},
}

class PerformanceTracker:
    """Class to track performance metrics of LLM calls"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = []
        self.metrics_file = METRICS_DIR / f"{model_name.replace('/', '_')}_metrics.json"
        
        # Load existing metrics if file exists
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except json.JSONDecodeError:
                self.metrics = []
    
    def record_metric(self, operation, prompt_tokens, completion_tokens, latency, success, error=None):
        """Record a metric for an LLM operation"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "operation": operation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency,
            "success": success
        }
        
        if error:
            metric["error"] = str(error)
            
        self.metrics.append(metric)
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return metric
    
    def get_summary(self):
        """Get a summary of the metrics"""
        if not self.metrics:
            return {"model": self.model_name, "count": 0}
        
        successful_metrics = [m for m in self.metrics if m["success"]]
        
        if not successful_metrics:
            return {
                "model": self.model_name,
                "count": len(self.metrics),
                "success_rate": 0,
                "error_count": len(self.metrics)
            }
        
        total_prompt_tokens = sum(m["prompt_tokens"] for m in successful_metrics)
        total_completion_tokens = sum(m["completion_tokens"] for m in successful_metrics)
        total_tokens = total_prompt_tokens + total_completion_tokens
        avg_latency = sum(m["latency_seconds"] for m in successful_metrics) / len(successful_metrics)
        
        # Estimate cost based on model (updated July 2025)
        # Default values
        cost_per_1k_prompt = MODEL_PRICING["default"]["input"]
        cost_per_1k_completion = MODEL_PRICING["default"]["output"]
        
        # Get pricing for the model
        if self.model_name in MODEL_PRICING:
            pricing = MODEL_PRICING[self.model_name]
            cost_per_1k_prompt = pricing["input"]
            cost_per_1k_completion = pricing["output"]
        # Fallback to partial matching if exact match not found
        else:
            for model_key, pricing in MODEL_PRICING.items():
                if model_key != "default" and model_key in self.model_name:
                    cost_per_1k_prompt = pricing["input"]
                    cost_per_1k_completion = pricing["output"]
                    break
        
        estimated_cost = (total_prompt_tokens / 1000 * cost_per_1k_prompt) + \
                         (total_completion_tokens / 1000 * cost_per_1k_completion)
        
        return {
            "model": self.model_name,
            "count": len(self.metrics),
            "success_count": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics),
            "error_count": len(self.metrics) - len(successful_metrics),
            "avg_latency_seconds": avg_latency,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_call": total_tokens / len(successful_metrics),
            "estimated_cost_usd": estimated_cost
        }

# Let's use a simpler approach with the OpenAI ChatModel
class OpenRouterChatModel(ChatOpenAI):
    """Chat model that uses OpenRouter API"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        # Get API key from environment if not provided
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Initialize with OpenAI's ChatModel but override the API base URL
        super().__init__(
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )

def get_chat_llm(model_name="gpt-4.1-mini", temperature=0):
    """
    Get a chat LLM model with performance tracking
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for the model
        
    Returns:
        A wrapped LLM model with performance tracking
    """
    # Get model info
    model_info = MODEL_INFO.get(model_name, {"provider": "openai", "type": "chat"})
    tracker = PerformanceTracker(model_name)
    
    # Create the base model
    if model_info["provider"] == "openrouter":
        base_model = OpenRouterChatModel(
            model=model_name,
            temperature=temperature
        )
    else:  # Default to OpenAI
        base_model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
    
    # Create a wrapper class to track performance
    class TrackedLLM:
        def __init__(self, llm, tracker):
            self.llm = llm
            self.tracker = tracker
        
        def invoke(self, messages):
            start_time = time.time()
            success = True
            error = None
            prompt_tokens = 0
            completion_tokens = 0
            
            try:
                response = self.llm.invoke(messages)
                
                # Try to get token usage from response
                if hasattr(response, 'response_metadata') and response.response_metadata:
                    metadata = response.response_metadata
                    if 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        prompt_tokens = token_usage.get('prompt_tokens', 0)
                        completion_tokens = token_usage.get('completion_tokens', 0)
                
                # For our custom OpenRouter implementation
                if hasattr(response, 'llm_output') and response.llm_output:
                    if 'token_usage' in response.llm_output:
                        token_usage = response.llm_output['token_usage']
                        prompt_tokens = token_usage.get('prompt_tokens', 0)
                        completion_tokens = token_usage.get('completion_tokens', 0)
                
                latency = time.time() - start_time
                self.tracker.record_metric(
                    "invoke", 
                    prompt_tokens, 
                    completion_tokens, 
                    latency, 
                    success
                )
                return response
            except Exception as e:
                success = False
                error = str(e)
                latency = time.time() - start_time
                self.tracker.record_metric(
                    "invoke", 
                    prompt_tokens, 
                    completion_tokens, 
                    latency, 
                    success, 
                    error
                )
                raise e
    
    return TrackedLLM(base_model, tracker)

def get_openai_llm(model_name="text-davinci-003", temperature=0):
    """Get an OpenAI completion model with performance tracking"""
    tracker = PerformanceTracker(model_name)
    base_model = OpenAI(model_name=model_name, temperature=temperature)
    
    # Create a wrapper class to track performance
    class TrackedLLM:
        def __init__(self, llm, tracker):
            self.llm = llm
            self.tracker = tracker
        
        def invoke(self, prompt):
            start_time = time.time()
            success = True
            error = None
            prompt_tokens = 0
            completion_tokens = 0
            
            try:
                response = self.llm.invoke(prompt)
                
                # Try to get token usage from response
                if hasattr(response, 'response_metadata') and response.response_metadata:
                    metadata = response.response_metadata
                    if 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        prompt_tokens = token_usage.get('prompt_tokens', 0)
                        completion_tokens = token_usage.get('completion_tokens', 0)
                
                latency = time.time() - start_time
                self.tracker.record_metric(
                    "invoke", 
                    prompt_tokens, 
                    completion_tokens, 
                    latency, 
                    success
                )
                return response
            except Exception as e:
                success = False
                error = str(e)
                latency = time.time() - start_time
                self.tracker.record_metric(
                    "invoke", 
                    prompt_tokens, 
                    completion_tokens, 
                    latency, 
                    success, 
                    error
                )
                raise e
    
    return TrackedLLM(base_model, tracker)

def get_openai_embeddings(model="text-embedding-3-small"):
    """Get OpenAI embeddings"""
    return OpenAIEmbeddings(model=model)

def get_model_metrics_summary():
    """Get a summary of metrics for all models"""
    summaries = []
    
    for file in METRICS_DIR.glob("*_metrics.json"):
        model_name = file.stem.replace("_metrics", "").replace("_", "/")
        tracker = PerformanceTracker(model_name)
        summaries.append(tracker.get_summary())
    
    return summaries

def calculate_cost(model_name, prompt_tokens, completion_tokens):
    """Calculate the cost of a request based on token usage"""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    
    input_cost = prompt_tokens / 1000 * pricing["input"]
    output_cost = completion_tokens / 1000 * pricing["output"]
    
    return input_cost, output_cost, input_cost + output_cost