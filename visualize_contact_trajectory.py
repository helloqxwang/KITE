"""Explore curated source lifting motions and autoregressive target decoding."""
from visualization.trajectory_viewer import main

if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
