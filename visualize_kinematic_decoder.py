"""Browse synthetic interaction intents and select an embodiment in the browser."""
from visualization.decoder_examples import main

if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
