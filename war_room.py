"""
War Room (CrewAI-enabled, smarter)

This version runs actual multi-agent debate if you have:
- Python 3.12 venv (venv-warroom)
- crewai + crewai-tools installed
- OPENAI_API_KEY set

If OPENAI_API_KEY is missing, it prints a structured fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

FALLBACK_TEXT = """
WAR ROOM FALLBACK (No API key detected)
- Please set OPENAI_API_KEY to run real agent debate.
- This file supports a rigorous, multi-round debate with scoring rubric.
"""


@dataclass
class StrategyScore:
    ev_after_fees: float
    capacity_usd_per_day: float
    tail_risk: float
    latency_requirement_ms: int
    operational_complexity: float

    def total(self) -> float:
        return (
            10.0 * self.ev_after_fees
            + 0.00001 * self.capacity_usd_per_day
            - 5.0 * self.tail_risk
            - 0.001 * self.latency_requirement_ms
            - 2.0 * self.operational_complexity
        )


EVIDENCE_PACK = """
Evidence constraints (from your repo + data notes):
- Fees/slippage are the biggest killers; need >=2–3% edge after fees.
- Basket/negative-risk arbs are structural if mutually exclusive sets dislocate.
- Market making can scale but needs hedging/latency control.
- 15m crypto markets are fast; avoid one-leg exposure.
"""


def run_crewai_debate() -> str:
    from crewai import Agent, Task, Crew, Process

    quant = Agent(
        role="Aggressive Quant",
        goal="Find a high-capacity 15m Polymarket strategy that can reach $8k/day.",
        backstory=(
            "Former HFT. You care about capacity, speed, and measurable edge."
        ),
        verbose=True,
    )

    risk = Agent(
        role="Risk Manager",
        goal="Kill unsafe ideas; enforce hard constraints and tail-risk controls.",
        backstory=(
            "You only approve strategies with real edge after fees and "
            "minimized one-leg exposure."
        ),
        verbose=True,
    )

    strategist = Agent(
        role="Chief Strategist",
        goal="Select the winning strategy using a scoring rubric and produce bot-ready rules.",
        backstory=(
            "You synthesize ideas into an executable plan with entry/exit, "
            "sizing, and kill-switches."
        ),
        verbose=True,
    )

    task1 = Task(
        description=(
            "Round 1: Propose three candidate strategies. For each, estimate "
            "EV after fees, capacity, tail risk, and latency needs. Use this evidence:\n"
            f"{EVIDENCE_PACK}"
        ),
        agent=quant,
    )

    task2 = Task(
        description=(
            "Round 2: Attack each candidate with concrete failure modes: "
            "fees, slippage, one-leg fill, liquidity fragmentation, and "
            "resolution risk. Reject weak ideas."
        ),
        agent=risk,
    )

    task3 = Task(
        description=(
            "Round 3: Select the best candidate, score it with the rubric, "
            "and output a bot-ready spec (entry/exit, sizing, kill-switches)."
        ),
        agent=strategist,
    )

    crew = Crew(
        agents=[quant, risk, strategist],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print(FALLBACK_TEXT)
    else:
        print(run_crewai_debate())
