from concurrent.futures import Executor
from threading import Thread, Condition, RLock
from time import time, sleep
import random


class ModuleScheduler:
    def __init__(self, func: callable, workers: Executor, max_overlaps: int,
                 input_producer: callable, output_consumer: callable,
                 target_frequency: float):
        self.workers = workers
        self.max_overlaps = max_overlaps
        self.input_producer = input_producer
        self.output_consumer = output_consumer
        self.func = func
        self.frequency = target_frequency
        self.target_frequency = target_frequency
        self.working = False
        self.alive = True
        self.working_status = Condition()

        tstart = time()
        self.run_func(target_time=tstart)
        tend = time()
        self.avg_exec_time = tend - tstart
        self.exp_exec_time = tend - tstart
        self.movement = 0
        self.movement_direction = 0
        self.num_calls = 1
        self.exec_time_lock = RLock()

        Thread(target=self.schedule_loop, daemon=True).start()

    def run_func(self, target_time):
        args, kwargs = self.input_producer(target_time)
        self.output_consumer(time=target_time,
                             input=(args, kwargs),
                             output=self.func(*args, **kwargs))

    def timed_run(self):
        tstart = time()
        self.run_func(target_time=tstart)
        elapsed_time = time() - tstart
        with self.exec_time_lock:
            avg_coeff = 2 / (self.num_calls + 2)
            exp_coeff = 3 / (self.num_calls + 3)
            self.avg_exec_time = self.avg_exec_time * (1 - avg_coeff) + elapsed_time * avg_coeff
            self.exp_exec_time = self.exp_exec_time * (1 - exp_coeff) + elapsed_time * exp_coeff
            self.movement = abs(self.exp_exec_time - self.avg_exec_time) / \
                            max(self.exp_exec_time, self.avg_exec_time)
            self.movement **= 0.3
            self.num_calls += 1
            # print(self.avg_exec_time)

    def schedule_loop(self):
        while self.alive:
            while self.working:
                period = 1 / self.frequency
                if self.avg_exec_time / (period * (1+self.movement)) > self.max_overlaps:
                    period = 1.05 * self.avg_exec_time / self.max_overlaps
                    self.frequency = 1 / (period * (1+self.movement))
                self.workers.submit(fn=self.timed_run)
                sleep(period)
                if random.random() < self.movement:
                    sleep(period)
            with self.working_status:
                self.working_status.wait_for(predicate=lambda: self.working or not self.alive)
        self.workers.shutdown()

    def start(self):
        with self.working_status:
            self.working = True
            self.working_status.notify()

    def stop(self):
        with self.working_status:
            self.working = False

    def shutdown(self):
        with self.working_status:
            self.working = False
            self.alive = False


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor


    def func(x):
        for i in range(1000000):
            b = x ** 2
        return x ** 2


    counter = 0


    def in_prod(time):
        global counter
        c = counter
        counter += 1
        return [c], {}


    def out_cons(time, input, output):
        print("IN: %s | OUT: %s | TIME: %s" % (str(input), str(output), str(time)))


    class Ctrl:
        def __init__(self):
            pass

        def notify_frequency(self, f):
            print("notified %s" % str(f))


    mod = ModuleScheduler(func=func,
                          workers=ThreadPoolExecutor(max_workers=10),
                          max_overlaps=10,
                          input_producer=in_prod,
                          output_consumer=out_cons,
                          target_frequency=2)

    mod.start()

    sleep(100)
