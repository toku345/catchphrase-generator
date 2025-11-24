from typing import Optional

# Maximum length for background field in characters
MAX_BACKGROUND_LENGTH = 2048


class CatchphraseRequest:
    """Catchphrase generation request model"""

    def __init__(
        self,
        starting_word: Optional[str] = None,
        background: Optional[str] = None,
        style: Optional[str] = None
    ):
        self.starting_word = starting_word
        self.background = background
        self.style = style

    @classmethod
    def from_dict(cls, data: dict) -> "CatchphraseRequest":
        """Create request from dictionary"""
        return cls(
            starting_word=data.get("starting_word"),
            background=data.get("background"),
            style=data.get("style")
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate request parameters
        Returns: (is_valid, error_message)
        """
        # Check if background is provided
        if self.background is None:
            return False, "background is required"

        # Check if background is a string
        if not isinstance(self.background, str):
            return False, "background must be a string"

        # Check if background is not empty after stripping whitespace
        if not self.background.strip():
            return False, "background cannot be empty"

        # Check if background length is within limits
        if len(self.background) > MAX_BACKGROUND_LENGTH:
            return False, f"background exceeds maximum length of {MAX_BACKGROUND_LENGTH} characters"

        return True, None


class CatchphraseResponse:
    """Catchphrase generation response model"""

    def __init__(self, catchphrase: str, model_used: str):
        self.catchphrase = catchphrase
        self.model_used = model_used

    def to_dict(self) -> dict:
        """Convert response to dictionary"""
        return {
            "catchphrase": self.catchphrase,
            "model_used": self.model_used
        }
