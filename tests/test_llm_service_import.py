import sys

def test_llm_service_import():
    print('sys.path:', sys.path)
    try:
        import forest_app.integrations.llm_service
        print('Import succeeded')
    except Exception as e:
        print('Import failed:', e)
        raise
