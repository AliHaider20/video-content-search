import logging


LOGGER_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(verbose: bool = False, force: bool = False) -> None:
    """Configure predictable process-wide logging for CLI and service entry points."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOGGER_FORMAT,
        force=force,
    )
