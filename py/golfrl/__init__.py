"""Python trainer for the Sunshine Golf Classic agent.

The environment is the Rust port of the game (golfsim/, bit-identical to the
JS game), loaded through ctypes; the policy is PyTorch and trains on the GPU
when there is one. Checkpoints are written in the JS format, so the browser
agent (rl/web/) and the JS tools load them unchanged.
"""
