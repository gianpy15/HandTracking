from library.utils.logging import log, WARNINGS
import types

STD_DEPRECATED_CLASS_MSG = "Class %s has been deprecated. Please do not use it."
DEPRECATED_CLASS_ALTERNATIVE_FORMAT_GENERIC_ALT = "Use %s instead"
DEPRECATED_CLASS_ALTERNATIVE_FORMAT_FUNC_ALT = "Call %s instead"


def deprecated_class(cla=None, *, message=STD_DEPRECATED_CLASS_MSG,
                     alternative=None):
    def _class_deco(cl):
        class dep(cl):
            def __init__(self, *args, **kwargs):
                log(message % cl.__name__, level=WARNINGS)
                if alternative is not None:
                    if isinstance(alternative, types.FunctionType):
                        log(DEPRECATED_CLASS_ALTERNATIVE_FORMAT_FUNC_ALT % alternative.__name__,
                            level=WARNINGS)
                    elif hasattr(alternative, '__name__'):
                        log(DEPRECATED_CLASS_ALTERNATIVE_FORMAT_GENERIC_ALT % alternative.__name__,
                            level=WARNINGS)
                    else:
                        log(DEPRECATED_CLASS_ALTERNATIVE_FORMAT_GENERIC_ALT % str(alternative),
                            level=WARNINGS)

                cl.__init__(self, *args, **kwargs)

        return dep
    if cla is None:
        return _class_deco
    return _class_deco(cla)


@deprecated_class
class foo:
    def __init__(self):
        print("foo!")

if __name__ == '__main__':
    a = foo()
