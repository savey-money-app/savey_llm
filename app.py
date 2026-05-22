"""Compatibility entry point for the queue worker."""
import asyncio

from worker import main

if __name__ == "__main__":
    asyncio.run(main())
