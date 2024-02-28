from mdagent.utils.clear_mem import ClearMemory


def cleanup_tests():
    clear_mem = ClearMemory()
    clear_mem.clear_ckpts_root()


def pytest_sessionfinish(session, exitstatus):
    cleanup_tests()
