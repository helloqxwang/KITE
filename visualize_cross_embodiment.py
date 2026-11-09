"""Compare realization of the same latent intent across target embodiments."""
from visualization.cross_embodiment import main

if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
