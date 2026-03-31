from dataclasses import dataclass
import asyncio
import litellm


@dataclass
class Persona:
    name: str
    description: str

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, one of the three MAGI decision nodes.\n"
            f"Your perspective: {self.description}\n"
            "Analyze the following query independently. Provide your honest assessment."
        )


# Built-in personas
MELCHIOR = Persona("Melchior", "You think like an analytical scientist. Prioritize logic, evidence, and precision.")
BALTHASAR = Persona("Balthasar", "You think like an empathetic caregiver. Prioritize human impact, safety, and ethical considerations.")
CASPER = Persona("Casper", "You think like a pragmatic realist. Prioritize feasibility, efficiency, and practical outcomes.")


class MagiNode:
    def __init__(self, name: str, model: str, persona: Persona, timeout: float = 60.0):
        self.name = name
        self.model = model
        self.persona = persona
        self.timeout = timeout

    async def query(self, prompt: str) -> str:
        """Send a query to this node's LLM. Returns the response text or raises on failure."""
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.persona.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    num_retries=3,
                ),
                timeout=self.timeout,
            )
            msg = response.choices[0].message
            content = msg.content
            # Reasoning models (e.g. MiniMax M2.7) put output in reasoning_content
            if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                content = msg.reasoning_content
            if not content or not content.strip():
                raise ValueError(f"Node {self.name} returned empty response")
            return content.strip()
        except asyncio.TimeoutError:
            print(f"Node {self.name} ({self.model}) TIMEOUT after {self.timeout}s")
            raise TimeoutError(f"Node {self.name} ({self.model}) timed out after {self.timeout}s")
        except litellm.AuthenticationError as e:
            print(f"Node {self.name} ({self.model}) AUTH ERROR: {e}")
            raise AuthenticationError(
                f"Node {self.name} authentication failed. "
                f"Please set the API key for {self.model}. Error: {e}"
            ) from e
        except Exception as e:
            print(f"Node {self.name} ({self.model}) ERROR: {type(e).__name__}: {e}")
            raise e


class AuthenticationError(Exception):
    pass
