"""Explore a fixed latent intent under changes to the initial configuration."""
from visualization.initialization import main

if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
