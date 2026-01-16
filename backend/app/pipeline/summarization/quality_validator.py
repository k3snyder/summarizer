"""
Quality validation logic for summarization.

Matches the quality loop behavior from summarizer.py:
- 30 max attempts with decreasing quality threshold
- Threshold: 90% for attempts 1-5, 85% for attempts 6+
- Model tier rotation: tier 1 (1-10), tier 2 (11-20), tier 3 (21-30)
- Prompt key: primary for attempts 1-5, fallback for attempts 6+
"""


class QualityValidator:
    """Validates summary quality and determines retry strategy.

    Encapsulates the quality loop logic from summarizer.py into a
    testable class with clear methods for threshold, tier, and prompt
    selection based on attempt number.
    """

    def __init__(
        self,
        threshold_high: int = 90,
        threshold_low: int = 85,
        tier_1_max: int = 10,
        tier_2_max: int = 20,
        prompt_switch_at: int = 5,
    ):
        """Initialize quality validator.

        Args:
            threshold_high: Quality threshold for first attempts (default: 90%)
            threshold_low: Quality threshold after initial attempts (default: 85%)
            tier_1_max: Max attempt for tier 1 model (default: 10)
            tier_2_max: Max attempt for tier 2 model (default: 20)
            prompt_switch_at: Attempt number to switch to fallback prompt (default: 5)
        """
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.tier_1_max = tier_1_max
        self.tier_2_max = tier_2_max
        self.prompt_switch_at = prompt_switch_at

    def get_threshold_for_attempt(self, attempt: int) -> int:
        """Get the quality threshold for a given attempt number.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Quality threshold as percentage (e.g., 90 for 90%)
        """
        if attempt <= self.prompt_switch_at:
            return self.threshold_high
        return self.threshold_low

    def get_model_tier_for_attempt(self, attempt: int) -> int:
        """Get the model tier for a given attempt number.

        Tier rotation:
        - Attempts 1-10: Tier 1 (primary model)
        - Attempts 11-20: Tier 2 (secondary model)
        - Attempts 21-30: Tier 3 (tertiary model)

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Model tier number (1, 2, or 3)
        """
        if attempt <= self.tier_1_max:
            return 1
        elif attempt <= self.tier_2_max:
            return 2
        else:
            return 3

    def get_prompt_key_for_attempt(self, attempt: int) -> str:
        """Get the prompt key for a given attempt number.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            "primary" for first attempts, "fallback" for later attempts
        """
        if attempt <= self.prompt_switch_at:
            return "primary"
        return "fallback"

    def validate(self, relevancy: float, attempt: int) -> bool:
        """Check if a relevancy score passes the quality threshold.

        Args:
            relevancy: Relevancy score as percentage (e.g., 92.5)
            attempt: Current attempt number (1-indexed)

        Returns:
            True if relevancy meets or exceeds threshold, False otherwise
        """
        threshold = self.get_threshold_for_attempt(attempt)
        return relevancy >= threshold
