"""The `prompt` stage's building blocks: choose a keyframe and an object to prompt the mask stage with, on raw
video that carries no label."""

from fastdo.prompt.choose import (
    HAND_CONTACT_OBJECT,
    PromptChoice,
    PromptThresholds,
    candidate_rows,
    choose_prompt,
    mask_is_plausible,
)

from fastdo.prompt.hands23_client import Hands23Client, Hands23Error, Hands23Result

__all__ = ["HAND_CONTACT_OBJECT", "Hands23Client", "Hands23Error", "Hands23Result", "PromptChoice",
           "PromptThresholds", "candidate_rows", "choose_prompt", "mask_is_plausible"]
