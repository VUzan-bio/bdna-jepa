"""Integration test: single training step on tiny data.

Test cases:
    - Build tiny model (2L×64D) + 100-sample dataset
    - One forward+backward pass completes without error
    - Loss is finite and has grad_fn
    - EMA update doesn't crash
    - All 4 loss components are returned in output dict
"""
# TODO: Implement integration test with tiny model
