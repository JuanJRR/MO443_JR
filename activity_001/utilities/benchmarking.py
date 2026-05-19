import logging

logger = logging.getLogger(__name__)


def benchmarking(func:callable):
    """
    Decorator to measure the execution time of a function.

    This utility calculates the elapsed time during the execution of a 
    specific function using a high-resolution timer. The result is 
    automatically reported through the logging system, facilitating 
    performance analysis for image processing algorithms.

    Args:
        func (Callable): The function or method to be benchmarked.

    Returns:
        Callable: A wrapper function that executes the original function, 
            logs its execution time, and returns its original result.
       
    """

    def wrapper(*args, **kwargs):
        import time

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        report = f'''
--- Benchmarking: {func.__name__} ---
  --Tiempo: {end_time - start_time:.6f}s
'''
        logger.info(report)
        return result

    return wrapper
