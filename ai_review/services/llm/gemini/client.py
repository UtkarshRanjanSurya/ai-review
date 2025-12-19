from ai_review.clients.gemini.client import get_gemini_http_client
from ai_review.clients.gemini.schema import (
    GeminiPartSchema,
    GeminiContentSchema,
    GeminiChatRequestSchema,
    GeminiGenerationConfigSchema,
)
from ai_review.config import settings
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema


class GeminiLLMClient(LLMClientProtocol):
    def __init__(self):
        self.http_client = get_gemini_http_client()

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        # 🔴 IMPORTANT: Gemini v1 does NOT support system_instruction
        # Merge system + user prompt into a single user message
        merged_prompt = f"{prompt_system}\n\n{prompt}"

        request = GeminiChatRequestSchema(
            contents=[
                GeminiContentSchema(
                    role="user",
                    parts=[GeminiPartSchema(text=merged_prompt)],
                )
            ],
            generation_config=GeminiGenerationConfigSchema(
                temperature=settings.llm.meta.temperature,
                max_output_tokens=settings.llm.meta.max_tokens,
            ),
        )

        response = await self.http_client.chat(request)

        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

