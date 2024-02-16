import time

from app.launcher import Launcher

if __name__ == "__main__":
    start_time = time.perf_counter()
    Launcher.start()
    print(f"Execution time: {time.perf_counter() - start_time} seconds")
