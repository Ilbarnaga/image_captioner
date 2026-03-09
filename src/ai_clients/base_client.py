"""
Base AI Client Module.

This module defines the abstract base class `AsyncVisionClient`, which sets
the blueprint for all specific API provider clients (e.g., OpenAI, Claude).
It enforces a uniform interface for configuring and interacting with vision models.
"""

from abc import ABC, abstractmethod


class AsyncVisionClient(ABC):
    """
    Abstract Base Class enforcing the contract for any Vision API Client.

    This class ensures that all child clients share common configuration
    properties and implement the required `generate_caption` asynchronous method.
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