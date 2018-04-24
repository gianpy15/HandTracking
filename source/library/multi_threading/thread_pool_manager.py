from concurrent.futures import ThreadPoolExecutor
import time


class ThreadPoolManager:
    __thread_pool = None

    @staticmethod
    def get_thread_pool():
        ThreadPoolManager.__thread_pool = ThreadPoolManager.__thread_pool or ThreadPoolExecutor()
        return ThreadPoolManager.__thread_pool


if __name__ == '__main__':
    def foo(x, y):
        time.sleep(1)
        print(x + y - y)

    p = ThreadPoolManager.get_thread_pool()
    for i in range(1000):
        p.submit(foo, i, i+1)
