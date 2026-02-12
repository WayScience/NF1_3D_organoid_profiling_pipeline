class ProcessorTypeError(Exception):
    """
    Exception raised when an unrecognized processor type is encountered.
    Use 'CPU' or 'GPU' as valid processor types.
    """

    def __str__(self):
        """Return a standardized error message."""
        return "Processor type not recognized. Use 'CPU' or 'GPU'. "
