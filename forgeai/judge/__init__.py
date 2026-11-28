"""The judge (spec §6.2-6.4): deterministic layer 1, LLM layer 2, canaries, scorecards.

Nothing here knows a domain. ``store`` is the only module an optimizer may import; ``sealed`` holds the
test split and nothing on the optimizer's import path may reach it (enforced by ``tests/test_sealed.py``).
"""
