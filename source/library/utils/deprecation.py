from library.utils.logging import log, WARNINGS
import types

STD_DEPRECATED_CLASS_MSG = "Class %s has been deprecated. Please do not use it."
STD_DEPRECATED_FUNC_MSG = "Function %s has been deprecated. Please do not use it."
DEPRECATED_ALTERNATIVE_FORMAT_GENERIC_ALT = "Use %s instead"
DEPRECATED_ALTERNATIVE_FORMAT_FUNC_ALT = "Call %s instead"


def deprecated_class(cla=None, *, message=STD_DEPRECATED_CLASS_MSG,
                     alternative=None):
    def _class_deco(cl):
        class dep(cl):
            def __init__(self, *args, **kwargs):
                log(message % cl.__name__, level=WARNINGS)
                if alternative is not None:
                    if isinstance(alternative, types.FunctionType):
                        log(DEPRECATED_ALTERNATIVE_FORMAT_FUNC_ALT % alternative.__name__,
                            level=WARNINGS)
                    elif hasattr(alternative, '__name__'):
                        log(DEPRECATED_ALTERNATIVE_FORMAT_GENERIC_ALT % alternative.__name__,
                            level=WARNINGS)
                    else:
                        log(DEPRECATED_ALTERNATIVE_FORMAT_GENERIC_ALT % str(alternative),
                            level=WARNINGS)

                cl.__init__(self, *args, **kwargs)

        return dep
    if cla is None:
        return _class_deco
    return _class_deco(cla)


def deprecated_fun(fun=None, *, message=STD_DEPRECATED_FUNC_MSG,
                     alternative=None):
    def _fun_deco(f):
        def dep(*args, **kwargs):
            log(message % f.__name__, level=WARNINGS)
            if alternative is not None:
                if isinstance(alternative, types.FunctionType):
                    log(DEPRECATED_ALTERNATIVE_FORMAT_FUNC_ALT % alternative.__name__,
                        level=WARNINGS)
                elif hasattr(alternative, '__name__'):
                    log(DEPRECATED_ALTERNATIVE_FORMAT_GENERIC_ALT % alternative.__name__,
                        level=WARNINGS)
                else:
                    log(DEPRECATED_ALTERNATIVE_FORMAT_GENERIC_ALT % str(alternative),
                        level=WARNINGS)

            return f(*args, **kwargs)

        return dep
    if fun is None:
        return _fun_deco
    return _fun_deco(fun)


@deprecated_class
class foo:
    def __init__(self):
        print("foo!")

    def pri(self):
        print("fofofofofo")

@deprecated_fun(alternative=log)
def funfoo():
    print("funfoo!")
    return 0

if __name__ == '__main__':
    a = foo()
    a.pri()
    funfoo()
