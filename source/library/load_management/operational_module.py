from library.load_management.module_scheduler import ModuleScheduler
from library.load_management.frequency_decouple import Interpolator
from concurrent.futures import ThreadPoolExecutor


class NoOutputException(Exception):
    def __init__(self):
        Exception.__init__(self)


class OperationalModule:
    def __init__(self, func: callable, workers: int,
                 input_source: callable, output_adapter: callable,
                 working_frequency: float, controller=None,
                 interp_order=0, interp_samples=1):

        self.interpolator = Interpolator(order=interp_order, samples=interp_samples)
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
        try:
            self.interpolator[time] = self.output_adapter(input, output)
        except NoOutputException:
            pass

    def __getitem__(self, item):
        return self.interpolator[item]

    def __call__(self, time):
        return self[time]

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.stop()

    def shutdown(self):
        self.scheduler.shutdown()