import time

# Logs function runtime
def function_runtime(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time() - t1
        print(f'Runtime for {func.__name__}: {t2:.3f}s')
    return wrapper