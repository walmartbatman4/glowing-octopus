"""
unfold: cINN Unbinned Detector Unfolding package.
"""

import os
import tempfile

# Set local writable matplotlib cache dir inside temp directory
mpl_dir = os.path.join(tempfile.gettempdir(), "unfold_matplotlib_cache")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

__version__ = "0.1.0"
