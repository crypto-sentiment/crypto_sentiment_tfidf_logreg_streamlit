import time
from contextlib import contextmanager
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    print(get_project_root().absolute())
