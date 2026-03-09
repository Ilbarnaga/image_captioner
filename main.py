import asyncio
from src.core.cli import CaptionerCLI

if __name__ == "__main__":
    cli = CaptionerCLI()
    asyncio.run(cli.start())