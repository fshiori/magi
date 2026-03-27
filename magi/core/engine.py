from magi.core.node import MagiNode, Persona, MELCHIOR, BALTHASAR, CASPER
from magi.core.decision import Decision
from magi.protocols.vote import vote
from magi.protocols.critique import critique
from magi.trace.logger import TraceLogger
import os


class MAGI:
    """
    MAGI Disagreement OS — three LLMs, one decision.

    Usage:
        engine = MAGI()  # uses defaults from env or config
        decision = await engine.ask("What is quantum entanglement?")
        print(decision.ruling)
        print(decision.minority_report)
    """

    def __init__(
        self,
        melchior: str = "claude-sonnet-4-6",
        balthasar: str = "gpt-4o",
        casper: str = "gemini/gemini-2.5-pro",
        personas: tuple[Persona, Persona, Persona] | None = None,
        timeout: float = 30.0,
        trace_dir: str | None = None,
    ):
        p = personas or (MELCHIOR, BALTHASAR, CASPER)
        self.nodes = [
            MagiNode("melchior", melchior, p[0], timeout),
            MagiNode("balthasar", balthasar, p[1], timeout),
            MagiNode("casper", casper, p[2], timeout),
        ]
        self.trace_dir = trace_dir or os.path.expanduser("~/.magi/traces")
        self._logger = TraceLogger(self.trace_dir)

    async def ask(self, query: str, mode: str = "vote") -> Decision:
        """
        Ask MAGI a question. Returns a Decision with ruling, confidence,
        minority report, and full trace.

        Modes: "vote" (default), "critique", "adaptive", "escalate"
        Only "vote" is implemented in MVP.
        """
        if mode == "vote":
            decision = await vote(query, self.nodes)
            # 3-way split with no majority → auto-escalate to critique
            if decision.protocol_used == "vote_no_majority":
                decision = await critique(query, self.nodes)
        elif mode == "critique":
            decision = await critique(query, self.nodes)
        elif mode == "escalate":
            decision = await critique(query, self.nodes, max_rounds=2)
        else:
            raise NotImplementedError(f"Mode '{mode}' not yet implemented.")

        self._logger.log(decision)
        return decision
