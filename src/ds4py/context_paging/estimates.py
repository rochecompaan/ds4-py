"""Display-only estimates; never used as exact native request accounting."""

from ds4py.context_paging.text_units import utf16_length

TEXT_UNITS_PER_TOKEN = 4
IMAGE_TOKENS = 4_800


def estimate_text_tokens(text: str) -> int:
    return (utf16_length(text) + TEXT_UNITS_PER_TOKEN - 1) // TEXT_UNITS_PER_TOKEN


def estimate_image_tokens(count: int = 1) -> int:
    if count < 0:
        raise ValueError("image count must be non-negative")
    return count * IMAGE_TOKENS


__all__ = ["IMAGE_TOKENS", "TEXT_UNITS_PER_TOKEN", "estimate_image_tokens", "estimate_text_tokens"]
