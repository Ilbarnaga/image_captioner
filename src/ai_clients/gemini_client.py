"""
Google Gemini Client Module.

This module implements the `GeminiVisionClient` class, inheriting from
`AsyncVisionClient`. It structures multimodal and text requests natively 
for Google's Gemini models using the google-generativeai SDK.
"""

import google.generativeai as genai
from .base_client import AsyncVisionClient


class GeminiVisionClient(AsyncVisionClient):
    """
    Client for Google Gemini models (e.g., gemini-1.5-pro, gemini-1.5-flash).

    This class wraps the official Google Generative AI async SDK, handling 
    both multimodal image captioning and text-only semantic evaluations.
    """
    
    def __init__(
            self,
            api_key     : str,
            model       : str = "gemini-1.5-pro",
            max_tokens  : int = 300,
            temperature : float = 0.3
        ):
        """
        Initializes the Gemini client with authentication and generation configs.

        Args:
            api_key (str): The authentication key for the Google API.
            model (str, optional): The target AI model. Defaults to "gemini-1.5-pro".
            max_tokens (int, optional): The token generation limit. Defaults to 300.
            temperature (float, optional): The sampling temperature. Defaults to 0.3.
        """
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        
        # Configure the global Google API key
        genai.configure(api_key=api_key)
        
        # Instantiate the model instance and generation configurations
        self.client = genai.GenerativeModel(self.model)
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature
        )

    async def generate_caption(self, image_b64: str, prompt: str) -> str:
        """
        Generates a caption by dispatching a vision request to Gemini.

        Args:
            image_b64 (str): The base64 encoded string of the target image.
            prompt (str): The textual instruction block for the AI.

        Returns:
            str: The stripped textual response returned by Gemini.
        """
        # Gemini expects a dictionary with mime_type and the raw base64 data
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_b64
        }
        
        response = await self.client.generate_content_async(
            contents=[prompt, image_part],
            generation_config=self.generation_config
        )
        return response.text.strip()

    async def generate_text(self, prompt: str) -> str:
        """
        Generates a semantic text-only evaluation utilizing Gemini's text capabilities.

        Args:
            prompt (str): The evaluation rubric and text payload to be judged.

        Returns:
            str: The AI's structural judgment output.
        """
        response = await self.client.generate_content_async(
            contents=[prompt],
            generation_config=self.generation_config
        )
        return response.text.strip()