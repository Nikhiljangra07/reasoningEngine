"""
The five disciplines.

Pure-Python utilities that the engine calls during decision-making.
No LLM calls inside these modules — they compute on plain state. The
engine threads their outputs into prompts and uses them to weight or
filter findings.

  goal_supremacy        — claim vs real-goal discrimination,
                          real-goal recovery probe
  long_horizon          — projection across timeframes, compounding
                          vs decay signal
  opportunity_capture   — six-question test on a surfaced opening
  attachment_detection  — distortion patterns named without
                          diagnosing the user
  resource_conversion   — convertible-form evaluation on constraints,
                          criticism, sunk effort, dead-ends
"""

from src.identity.disciplines.goal_supremacy import (
    ServeScore,
    discriminate,
    surface_real_goal,
)
from src.identity.disciplines.long_horizon import (
    HorizonRead,
    HorizonSignal,
    compounding_signal,
    project,
)
from src.identity.disciplines.opportunity_capture import (
    CaptureVerdict,
    Opening,
    SIX_QUESTIONS,
    test,
)
from src.identity.disciplines.attachment_detection import (
    AttachmentFlag,
    AttachmentKind,
    scan,
)
from src.identity.disciplines.resource_conversion import (
    ConvertibilityScore,
    Resource,
    evaluate,
    latent_uses,
)

__all__ = [
    # goal_supremacy
    "ServeScore",
    "discriminate",
    "surface_real_goal",
    # long_horizon
    "HorizonRead",
    "HorizonSignal",
    "compounding_signal",
    "project",
    # opportunity_capture
    "CaptureVerdict",
    "Opening",
    "SIX_QUESTIONS",
    "test",
    # attachment_detection
    "AttachmentFlag",
    "AttachmentKind",
    "scan",
    # resource_conversion
    "ConvertibilityScore",
    "Resource",
    "evaluate",
    "latent_uses",
]
