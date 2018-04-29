import sys

DEBUG = 0  # expect to find the log spammed with debug info even from cycles
COMMENTARY = 1  # expect to find the log quite verbose with comments of what's going on, no cycles
WARNINGS = 2  # the log is quite clean, but tells all important stuff
IMPORTANT_WARNINGS = 3  # almost silent, just urgent notifications
ERRORS = 4  # dare speak only if something goes definitely wrong
SILENT = 5  # do not dare speak at all


VERBOSITY = WARNINGS
LOGFILE = sys.stdout


def set_verbosity(verb):
    """
    Set the current verbosity level of the logger. Enables filtering out all
    logging that has lower priority than the set level.
    :param verb: the current logging level, chosen between:
        DEBUG -> expect to find the log spammed with debug info even from cycles
        COMMENTARY -> expect to find the log quite verbose with comments of what's going on, no cycles
        WARNINGS -> the log is quite clean, but tells all important stuff
        IMPORTANT_WARNINGS -> almost silent, just urgent notifications
        ERRORS -> dare speak only if something goes definitely wrong
        SILENT -> do not dare speak at all

    """
    global VERBOSITY
    VERBOSITY = verb


def set_log_file(logfile):
    """
    Set all logging messages to be appended into a different file or file-like stream
    :param logfile: the stream where messages will be logged
    """
    global LOGFILE
    LOGFILE = logfile


def log(message, level=DEBUG):
    """
    Just output a message in the current log file if the current logging verbosity is greater
    than the one passed. Used for conditional logging.
    :param message: The message to be logged
    :param level: The priority to be given to the message
    """
    if level >= VERBOSITY:
        LOGFILE.write(str(message) + '\n')
        if level >= IMPORTANT_WARNINGS:
            LOGFILE.flush()
