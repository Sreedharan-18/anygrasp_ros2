"""
Stub eval_utils: disables Dex-Net-based evaluation so detection can run without heavy deps.
If anything ever calls into this, we fail loudly to make it obvious.
"""
def __getattr__(name):
    raise RuntimeError(
        "Dex-Net evaluation disabled. 'graspnetAPI.utils.eval_utils' is stubbed."
    )
