"""
Maths Layer 7: Game Theory (Strategic Multi-Agent Reasoning)

Physics models forces but not intentional actors. When the user's problem
involves another person making strategic moves — negotiations, relationship
dynamics, business competition — Game Theory handles it.

Activates when the manifold detects multiple intentional agents.

4 Operations:
1. Nash Equilibrium — stable states where no one changes strategy
2. Dominant Strategy — the best move regardless of what others do
3. Prisoner's Dilemma — cooperation traps
4. Zero-Sum vs Positive-Sum — reframing the game
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    Direction,
    FrameworkID,
    GameState,
    Problem,
    Variable,
)


def analyze_game(problem: Problem) -> GameState | None:
    """
    Run game theory analysis on the problem.

    Returns None if the problem doesn't involve multiple agents.
    Returns GameState with strategic analysis if it does.
    """
    # Detect agents from problem variables
    agents = _detect_agents(problem)

    if len(agents) < 2:
        return None  # not a multi-agent problem

    # Determine game type
    is_zero_sum = _detect_zero_sum(problem)

    # Find Nash equilibrium
    nash = _find_nash_equilibrium(problem, agents, is_zero_sum)

    # Find dominant strategies
    dominant = _find_dominant_strategies(problem, agents)

    # Check for prisoner's dilemma pattern
    is_pd = _detect_prisoners_dilemma(problem, agents)

    return GameState(
        agents=agents,
        is_zero_sum=is_zero_sum,
        nash_equilibrium=nash,
        dominant_strategies=dominant,
        prisoners_dilemma=is_pd,
    )


def extract_game_variables(game: GameState) -> list[Variable]:
    """Extract variables from game theory analysis for the manifold."""
    if game is None:
        return []

    variables = []

    # Zero-sum framing variable
    if game.is_zero_sum:
        variables.append(Variable(
            name="zero_sum_frame",
            description=(
                f"This is framed as a zero-sum game between {', '.join(game.agents)}. "
                "One party's gain = other's loss. "
                "Question: is it ACTUALLY zero-sum, or is the user framing it that way? "
                "If positive-sum options exist, the user is missing paths."
            ),
            magnitude=0.7,
            direction=Direction.NEGATIVE,
            confidence=0.6,
            source_framework=FrameworkID.GAME_THEORY,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Agents: {', '.join(game.agents)}",
                "Detected zero-sum framing in problem variables.",
            ],
        ))
    else:
        variables.append(Variable(
            name="positive_sum_opportunity",
            description=(
                f"Positive-sum game detected between {', '.join(game.agents)}. "
                "Both parties can gain. The optimal strategy involves "
                "finding the cooperative path, not the competitive one."
            ),
            magnitude=0.6,
            direction=Direction.POSITIVE,
            confidence=0.6,
            source_framework=FrameworkID.GAME_THEORY,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Agents: {', '.join(game.agents)}",
                "Positive-sum structure detected.",
            ],
        ))

    # Nash equilibrium variable
    if game.nash_equilibrium:
        variables.append(Variable(
            name="nash_equilibrium",
            description=game.nash_equilibrium,
            magnitude=0.7,
            direction=Direction.NEUTRAL,
            confidence=0.65,
            source_framework=FrameworkID.GAME_THEORY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                "Nash equilibrium: no player can improve by changing strategy alone.",
                "This may explain why the situation feels 'stuck' despite effort.",
            ],
        ))

    # Prisoner's dilemma variable
    if game.prisoners_dilemma:
        variables.append(Variable(
            name="prisoners_dilemma_trap",
            description=(
                f"Prisoner's dilemma pattern detected between {', '.join(game.agents)}. "
                "Both parties would benefit from cooperation, but the incentive "
                "structure pushes them to defect. Trust keeps breaking not because "
                "of bad intentions but because of bad structure."
            ),
            magnitude=0.8,
            direction=Direction.NEGATIVE,
            confidence=0.7,
            source_framework=FrameworkID.GAME_THEORY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                "Cooperation is optimal but individually irrational.",
                "Resolution requires structural change, not just communication.",
            ],
        ))

    # Dominant strategy variables
    for agent, strategy in game.dominant_strategies.items():
        variables.append(Variable(
            name=f"dominant_strategy_{agent}",
            description=f"Agent '{agent}' has a dominant strategy: {strategy}",
            magnitude=0.6,
            direction=Direction.NEUTRAL,
            confidence=0.6,
            source_framework=FrameworkID.GAME_THEORY,
            is_hidden=False,
            is_user_stated=False,
            evidence=[f"Dominant strategy for {agent}: {strategy}"],
        ))

    return variables


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_agents(problem: Problem) -> list[str]:
    """
    Detect intentional agents in the problem.

    Agents are identified by variables that reference other people,
    organizations, or entities making decisions.
    """
    agents = ["user"]  # the user is always an agent

    # Look for variables that reference other actors
    agent_keywords = [
        "partner", "cofounder", "boss", "manager", "team", "company",
        "competitor", "spouse", "parent", "friend", "colleague", "client",
        "investor", "board", "employee", "vendor", "landlord", "bank",
    ]

    for var in problem.variables:
        desc_lower = var.description.lower()
        name_lower = var.name.lower()
        combined = desc_lower + " " + name_lower

        for keyword in agent_keywords:
            if keyword in combined and keyword not in agents:
                agents.append(keyword)
                break

    # Also check the problem statement
    statement_lower = problem.statement.lower()
    for keyword in agent_keywords:
        if keyword in statement_lower and keyword not in agents:
            agents.append(keyword)

    return agents


def _detect_zero_sum(problem: Problem) -> bool:
    """
    Detect if the problem is framed as zero-sum.

    Zero-sum indicators:
    - Variables with opposing directions of similar magnitude
    - Language suggesting competition, winning/losing
    """
    positive_magnitude = sum(
        v.magnitude for v in problem.variables if v.direction == Direction.POSITIVE
    )
    negative_magnitude = sum(
        v.magnitude for v in problem.variables if v.direction == Direction.NEGATIVE
    )

    # If positive and negative forces are roughly balanced, it feels zero-sum
    if positive_magnitude > 0 and negative_magnitude > 0:
        ratio = min(positive_magnitude, negative_magnitude) / max(positive_magnitude, negative_magnitude)
        if ratio > 0.6:
            return True

    return False


def _find_nash_equilibrium(
    problem: Problem, agents: list[str], is_zero_sum: bool
) -> str | None:
    """
    Find Nash equilibrium — a stable state where no agent can
    improve by unilaterally changing strategy.

    In human problems, Nash equilibria explain why situations feel
    "stuck" — both parties are in a local optimum.
    """
    # Look for equilibrium-like patterns in variables
    stuck_indicators = [
        v for v in problem.variables
        if v.direction == Direction.NEUTRAL
        or (v.direction == Direction.NEGATIVE and v.magnitude < 0.4)
    ]

    positive_vars = [v for v in problem.variables if v.direction == Direction.POSITIVE]
    negative_vars = [v for v in problem.variables if v.direction == Direction.NEGATIVE]

    if stuck_indicators or (positive_vars and negative_vars):
        if is_zero_sum:
            return (
                f"Nash equilibrium detected: {', '.join(agents)} are locked in a "
                "competitive stable state. Neither can improve their position "
                "without the other also changing. Breaking this requires "
                "changing the game itself, not playing harder."
            )
        else:
            return (
                f"Nash equilibrium detected: {', '.join(agents)} have settled "
                "into a stable but suboptimal state. Both could do better "
                "through coordination, but unilateral moves make things worse. "
                "Resolution requires simultaneous strategy change."
            )

    return None


def _find_dominant_strategies(
    problem: Problem, agents: list[str]
) -> dict[str, str]:
    """
    Find dominant strategies — best move regardless of others' actions.

    For the user: look at their highest-confidence positive variable.
    For other agents: infer from their negative impact on the user.
    """
    strategies = {}

    # User's dominant strategy: their strongest positive lever
    user_positives = sorted(
        [v for v in problem.variables if v.direction == Direction.POSITIVE],
        key=lambda v: v.magnitude * v.confidence,
        reverse=True,
    )
    if user_positives:
        strategies["user"] = (
            f"Lean into '{user_positives[0].name}' — this is the strongest "
            f"lever (magnitude: {user_positives[0].magnitude:.2f})"
        )

    # Other agents' strategies: inferred from their impact
    for agent in agents:
        if agent == "user":
            continue
        # Find variables mentioning this agent
        agent_vars = [
            v for v in problem.variables
            if agent in v.name.lower() or agent in v.description.lower()
        ]
        if agent_vars:
            strongest = max(agent_vars, key=lambda v: v.magnitude)
            strategies[agent] = (
                f"Current strategy inferred from '{strongest.name}': "
                f"{strongest.description[:80]}"
            )

    return strategies


def _detect_prisoners_dilemma(
    problem: Problem, agents: list[str]
) -> bool:
    """
    Detect prisoner's dilemma pattern.

    Indicators:
    - Multiple agents with negative-direction variables (mutual harm)
    - High-magnitude positive variables that require cooperation
    - Trust-related language in the problem
    """
    if len(agents) < 2:
        return False

    # Check for mutual negative impact
    negative_vars = [v for v in problem.variables if v.direction == Direction.NEGATIVE]
    positive_vars = [v for v in problem.variables if v.direction == Direction.POSITIVE]

    # PD pattern: both sides losing (negatives) when cooperation (positives) is possible
    if len(negative_vars) >= 2 and len(positive_vars) >= 1:
        avg_negative_mag = sum(v.magnitude for v in negative_vars) / len(negative_vars)
        avg_positive_mag = sum(v.magnitude for v in positive_vars) / len(positive_vars)

        # If potential gains from cooperation exceed current losses
        if avg_positive_mag > avg_negative_mag * 0.8:
            return True

    return False
