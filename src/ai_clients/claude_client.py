"""
Anthropic Claude Client Module.

This module implements the `ClaudeVisionClient` class, inheriting from
`AsyncVisionClient`. It provides specific logic to format requests and
communicate with the Anthropic API for Claude models.
"""

from anthropic import AsyncAnthropic
from .base_client import AsyncVisionClient


class ClaudeVisionClient(AsyncVisionClient):
    """
    Client for Claude 3.5 Sonnet, Opus, and other Anthropic Vision APIs.

    This class wraps the official Anthropic async Python SDK, tailoring
    the payload structure specifically to Claude's required message format.
    """
    
    def __init__(
            self,
            api_key     : str,
            model       : str = "claude-3-5-sonnet-latest",
            max_tokens  : int = 300,
            temperature : float = 0.3
        ):
        """
        Initializes the Anthropic client with the required authentication and settings.

        Args:
            api_key (str): The API key for authenticating with Anthropic.
            model (str, optional): The Claude model to use. Defaults to "claude-3-5-sonnet-latest".
            max_tokens (int, optional): The token generation limit. Defaults to 300.
            temperature (float, optional): The sampling temperature. Defaults to 0.3.
        """
        # Initialize common properties inherited from the base class
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        
        # Instantiate the official Anthropic async client
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_caption(self, image_b64: str, prompt: str) -> str:
        """
        Generates a caption by dispatching a vision request to the Anthropic API.

        Args:
            image_b64 (str): The base64 string of the image.
            prompt (str): The textual instruction block for the AI.

        Returns:
            str: The stripped textual response returned by the Claude model.
        """
        # Format the payload according to Anthropic's multimodal API spec
        response = await self.client.messages.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                # Anthropic expects a pure base64 string without the 'data:image...' prefix
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Extract and return the raw text from the first content block
        return response.content[0].text.strip()