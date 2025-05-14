def test_backoff_import():
    import backoff
    print(f"Backoff version: {backoff.__version__}")
    print(f"Backoff file: {backoff.__file__}")
    assert hasattr(backoff, '__version__')
