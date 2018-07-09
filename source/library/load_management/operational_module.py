from library.load_management.module_scheduler import ModuleScheduler
from library.load_management.frequency_decouple import Interpolator
from concurrent.futures import ThreadPoolExecutor


class OperationalModule:
    INTERP_ORDER = 2
    INTERP_SAMPLES = 4

    def __init__(self, func: callable, workers: int,
                 input_source: callable, output_adapter: callable,
                 working_frequency: float, controller=None):

        self.interpolator = Interpolator(order=self.INTERP_ORDER, samples=self.INTERP_SAMPLES)
        self.executor_pool = ThreadPoolExecutor(max_workers=workers)
        self.output_adapter = output_adapter
        self.controller = controller

        self.scheduler = ModuleScheduler(func=func,
                                         workers=self.executor_pool,
                                         max_overlaps=workers,
                                         input_producer=input_source,
                                         output_consumer=self.feed_to_interpolator,
                                         working_frequency=working_frequency,
                                         controller=controller)

    def feed_to_interpolator(self, time, input, output):
        self.interpolator[time] = self.output_adapter(input, output)

    def __getitem__(self, item):
        return self.interpolator[item]

    def __call__(self, time):
        return self[time]