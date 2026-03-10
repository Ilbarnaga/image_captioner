"""
Base AI Client Module.

This module defines the abstract base class `AsyncVisionClient`, which sets
the blueprint for all specific API provider clients (e.g., OpenAI, Claude).
It enforces a uniform interface for configuring and interacting with both 
vision and text-only models.
"""

from abc import ABC, abstractmethod


class AsyncVisionClient(ABC):
    """
    Abstract Base Class enforcing the contract for any Vision API Client.

    This class ensures that all child clients share common configuration
    properties and implement the required asynchronous methods for generating
    both multimodal captions and text-only semantic evaluations.
    """
    
    def __init__(self, model: str, max_tokens: int, temperature: float):
        """
        Initializes the shared properties for the AI client.

        Args:
            model (str): The specific model identifier to be used (e.g., 'gpt-4o', 'claude-3-5-sonnet').
            max_tokens (int): The maximum number of tokens to generate in the response.
            temperature (float): The sampling temperature to control response randomness.
        """
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

    @abstractmethod
    async def generate_caption(self, image_b64: str, prompt: str) -> str:
        """
        Generates a caption using a single prompt and base64 encoded image.

        Args:
            image_b64 (str): The base64 string representation of the image.
            prompt (str): The textual instructions guiding the vision model.

        Returns:
            str: The generated caption text.
        """
        pass

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """
        Generates a text-only response without processing an image payload.
        
        This method is specifically utilized by the AI Judge in the Quality Control 
        pipeline to evaluate captions semantically without incurring the high token 
        costs associated with multimodal vision requests.

        Args:
            prompt (str): The textual instruction block for the AI (e.g., the Judge rubric).

        Returns:
            str: The text evaluation returned by the model.
        """
        pass
    