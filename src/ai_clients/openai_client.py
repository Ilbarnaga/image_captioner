"""
OpenAI-Compatible Client Module.

This module implements the `OpenAIVisionClient` class, inheriting from
`AsyncVisionClient`. It structures requests for OpenAI models (like GPT-4o)
or any compatible API endpoint (such as Grok or LocalLLMs).
"""

from openai import AsyncOpenAI
from .base_client import AsyncVisionClient


class OpenAIVisionClient(AsyncVisionClient):
    """
    Client for GPT-4o, Grok, and other OpenAI-compatible APIs.

    This class wraps the official OpenAI async Python SDK, handling both 
    multimodal image completion endpoints and standard text-only chat endpoints.
    """
    
    def __init__(
            self,
            api_key     : str,
            base_url    : str,
            model       : str = "gpt-4o",
            max_tokens  : int = 300,
            temperature : float = 0.3
        ):
        """
        Initializes the OpenAI-compatible client with authentication and routing.

        Args:
            api_key (str): The authentication key for the API.
            base_url (str): The custom or default endpoint URL (vital for Grok/xAI).
            model (str, optional): The target AI model. Defaults to "gpt-4o".
            max_tokens (int, optional): The token generation limit. Defaults to 300.
            temperature (float, optional): The sampling temperature. Defaults to 0.3.
        """
        # Initialize common properties inherited from the base class
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        
        # Instantiate the official OpenAI async client with custom base URL routing
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_caption(self, image_b64: str, prompt: str) -> str:
        """
        Generates a caption by dispatching a vision request to the OpenAI-compatible API.

        Args:
            image_b64 (str): The base64 encoded string of the target image.
            prompt (str): The textual instruction block for the AI.

        Returns:
            str: The stripped textual response returned by the target model.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # OpenAI requires the data URI scheme prefix for base64 images
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    async def generate_text(self, prompt: str) -> str:
        """
        Generates a semantic text-only evaluation utilizing the standard chat endpoint.

        Args:
            prompt (str): The evaluation rubric and text payload to be judged.

        Returns:
            str: The AI's structural judgment output.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()
    