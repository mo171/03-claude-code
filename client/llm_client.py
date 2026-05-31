from typing import Any, AsyncGenerator
import asyncio
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError, APIError
from client.response import StreamEventType, StreamEvent, TextDelta, TokenUsage


class LLMClient:
    def __init__(self) -> None:
        self.client: AsyncOpenAI | None = None
        self._max_retries: int =3

    def get_client(self) -> AsyncOpenAI:
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key="",
                base_url="https://openrouter.ai/api/v1",
            )
        return self.client

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def chat_completion(self, messages: list[dict[str, Any]], stream: bool = True) -> AsyncGenerator[StreamEvent, None]:
        
        for attempt in range(self._max_retries + 1):
            try:
                client = self.get_client()

                kwargs = {
                "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "messages": messages,
                "stream": stream,
            }

                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return 
            
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded: {e}",
                    )
                return
                
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API connection error: {e}",
                    )
                
            except APIStatusError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API status error: {e}",
                    )  
                return

            except APIError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API error: {e}",
                    )
                return 

            except Exception as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Unexpected error: {e}",
                )
                return                                          

    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)
        
        async for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    cached = 0
                    if getattr(chunk.usage, "prompt_tokens_details", None):
                        cached = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_tokens=cached,
                    )
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=None,
                        finish_reason=None,
                        usage=usage,
                    )
                continue
            
            choice = chunk.choices[0]
            text_delta = None
            if hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                text_delta = TextDelta(content=choice.delta.content)
            
            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                cached = 0
                if getattr(chunk.usage, "prompt_tokens_details", None):
                    cached = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=cached,
                )

            finish_reason = getattr(choice, "finish_reason", None)

            if text_delta or finish_reason or usage:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=text_delta,
                    finish_reason=finish_reason,
                    usage=usage,
                )

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)
            
        usage = None
        if response.usage:
            cached = 0
            if getattr(response.usage, "prompt_tokens_details", None):
                cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=cached,
            )
        
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )