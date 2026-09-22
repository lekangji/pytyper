"""US-QWERTY transition features supplied to, not enforced on, the network."""

from __future__ import annotations

KEYBOARD_FEATURES = 5

_ROWS = ("`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./")
_SHIFTED = dict(zip("~!@#$%^&*()_+{}|:\"<>?", "`1234567890-=[]\\;',./"))
_POSITIONS = {
    key: (column, row)
    for row, keys in enumerate(_ROWS)
    for column, key in enumerate(keys)
}


def _physical_key(character: str) -> str:
    if character in _SHIFTED:
        return _SHIFTED[character]
    return character.lower()


def transition_features(previous: str, upcoming: str) -> tuple[float, ...]:
    """Normalized geometry, hand switch, modifier, and boundary indicators."""

    prev_position = _POSITIONS.get(_physical_key(previous))
    next_position = _POSITIONS.get(_physical_key(upcoming))
    if prev_position is None or next_position is None:
        dx = dy = hand_switch = 0.0
    else:
        dx = min(1.0, abs(next_position[0] - prev_position[0]) / 11.0)
        dy = min(1.0, abs(next_position[1] - prev_position[1]) / 3.0)
        hand_switch = float((prev_position[0] < 5) != (next_position[0] < 5))
    shifted = float(upcoming.isalpha() and upcoming.isupper() or upcoming in _SHIFTED)
    boundary = float(upcoming.isspace() or previous.isspace())
    return dx, dy, hand_switch, shifted, boundary
