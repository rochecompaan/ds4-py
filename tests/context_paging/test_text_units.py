import pytest

from ds4py.context_paging.estimates import estimate_image_tokens, estimate_text_tokens
from ds4py.context_paging.text_units import clip_utf16, utf16_length, utf16_offset, utf16_slice


def test_utf16_lengths_and_offsets_count_astral_pairs_as_two_units() -> None:
    text = "a🙂b"

    assert utf16_length(text) == 4
    assert [utf16_offset(text, index) for index in range(4)] == [0, 1, 3, 4]
    assert utf16_slice(text, 1, 3) == "🙂"
    assert utf16_slice(text, 3) == "b"


def test_clips_never_leave_half_of_a_surrogate_pair() -> None:
    text = "abc🙂def"

    clipped, units = clip_utf16(text, 4)

    assert clipped == "abc"
    assert units == 3
    assert clip_utf16(text, 5) == ("abc🙂", 5)


def test_slices_avoid_splitting_pairs_at_either_boundary() -> None:
    assert utf16_slice("a🙂b", 2, 4) == "b"
    assert utf16_slice("a🙂b", 1, 2) == ""


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("", 0), ("abcd", 1), ("abcde", 2), ("🙂🙂", 1), ("🙂🙂🙂", 2)],
)
def test_display_text_estimates_use_four_utf16_units_per_token(text: str, tokens: int) -> None:
    assert estimate_text_tokens(text) == tokens


def test_display_image_estimate_is_fixed_and_independent_of_text() -> None:
    assert estimate_image_tokens() == 4800
    assert estimate_image_tokens(3) == 14_400
