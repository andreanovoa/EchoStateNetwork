"""Verbosity of echostatenetwork.

The package writes its reports (the hyperparameter search and the summary of ``train``, the report of each seed)
and its diagnostic messages (e.g. a network trained again, a training option changed by ``train``, the seeds
trained serially) to the logger ``'echostatenetwork'`` and its children, which are silent by default.
`set_verbose` shows them on stderr; an application that configures `logging` itself can set the level of the
``'echostatenetwork'`` logger instead (with both, its own handlers print every message a second time).

Warnings are not reports: an ignored keyword, a trained network that is not trained again, or fewer validation
folds than requested warn through `warnings`, whether or not `set_verbose` is on. A warning points at the line that
calls the package (`_warn`) and appears once per call: one call of `train` issues each warning once, however many
seeds, patches or evaluations of the search raise it, and in the calling process, so that the warnings filters and
`pytest.warns` of the caller apply. A filter that silences a warning of the package selects it by its message or
category, e.g. ``warnings.filterwarnings('ignore', message='(WFV|KFV): only')``: the module of a warning is the module
of the caller, not echostatenetwork.
"""

import contextlib
import contextvars
import logging
import os
import sys
import warnings

__all__ = ['set_verbose']

_HANDLER = 'echostatenetwork.verbose'
_PREFIX = '[echostatenetwork] '

# Frames of files in the package directory are internal to echostatenetwork.
_PACKAGE_PREFIX = os.path.dirname(__file__) + os.sep
# Inside `_warnings_once` or `_collected_warnings`: the pair (the warnings seen in the block, as (message, category)
# pairs; the list that collects them, or None when the block issues them). A context variable, not a global, so that
# threads that train at the same time do not share it.
_ONCE = contextvars.ContextVar('echostatenetwork_warnings_once', default=None)
# Libraries that call the package back: scikit-optimize calls the validation strategy in the search of `train`, through
# scikit-learn and joblib. Their frames between two frames of the package do not call into the package.
_CALLBACK_LIBRARIES = frozenset({'skopt', 'sklearn', 'joblib'})


def _caller(frame):
    """The frame that called into echostatenetwork, from `frame` outwards: the first frame that is neither in the
    package nor in a library that calls the package back (`_CALLBACK_LIBRARIES`); None if there is none."""
    while frame is not None:
        if not frame.f_code.co_filename.startswith(_PACKAGE_PREFIX):
            module = frame.f_globals.get('__name__') or ''
            if module.partition('.')[0] not in _CALLBACK_LIBRARIES:
                return frame
        frame = frame.f_back
    return None


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue a warning at the line that called into echostatenetwork: the first frame outside the package, as
    dynamodels does, skipping the frames of the libraries that call the package back (`_caller`).

    That line is the call of the user or, for a call through a subclass, a wrapper or a callback defined outside
    echostatenetwork (e.g. romda's ESN_model, or a validation strategy of the user that calls `compute_nMAE`), the
    line of that subclass, wrapper or callback. A fixed ``stacklevel`` cannot reach it, because the number of frames
    of the package in between varies: ``EchoStateNetwork(y, N_unit=50)`` passes through `__init__` and the helper
    that warns, whereas ``train`` with independent patches passes through more. The frames of scikit-optimize are
    skipped because it calls the validation strategy back in the search of `train`: the warning of a strategy of the
    package points at the line that calls `train`. Python shows a warning once per location by default, so a
    location inside the package would also merge the warnings of all callers into one. The frames are walked on
    every Python version: ``skip_file_prefixes`` (3.12 and later) cannot skip a library between two frames of the
    package.

    Inside `_warnings_once`, a warning that the block has issued already is dropped; inside `_collected_warnings`, it
    is collected instead of issued.
    """
    once = _ONCE.get()
    if once is not None:
        seen, collected = once
        key = (str(message), category)
        if key in seen:
            return
        seen.add(key)
        if collected is not None:
            collected.append(key)
            return
    caller = _caller(sys._getframe(1))
    if caller is None:                  # no frame outside the package called it
        warnings.warn(message, category, stacklevel=2)
        return
    # what warnings.warn does for the frame of the caller, so that the filters, the registry of the action 'default'
    # and pytest.warns apply; without module_globals, which Python 3.14 rejects with a DeprecationWarning for the
    # __main__ of a script. Globals without __name__ (code run by exec) give the module '<string>', as for
    # warnings.warn: warn_explicit drops a warning whose module is None.
    module = caller.f_globals
    warnings.warn_explicit(message, category, caller.f_code.co_filename, caller.f_lineno,
                           module=module.get('__name__') or '<string>',
                           registry=module.setdefault('__warningregistry__', {}))


@contextlib.contextmanager
def _warnings_once():
    """Issue every distinct warning of `_warn` once in the block, when it first occurs, e.g. once per call of `train`
    however many patches or seeds raise it. A block inside another block (or inside `_collected_warnings`) follows
    the outer block."""
    if _ONCE.get() is not None:
        yield
        return
    token = _ONCE.set((set(), None))
    try:
        yield
    finally:
        _ONCE.reset(token)


@contextlib.contextmanager
def _collected_warnings():
    """Collect the distinct warnings of `_warn` in the block, as ``(message, category)`` pairs in the list that it
    yields, without issuing them. A seed of ``train(n_seeds > 1)`` returns them to the parent process, which issues
    them: the process of a seed cannot reach the line of the caller, nor its warnings filters or `pytest.warns`."""
    collected = []
    token = _ONCE.set((set(), collected))
    try:
        yield collected
    finally:
        _ONCE.reset(token)


class _PrefixFormatter(logging.Formatter):
    """Prefix every line of a record, so that a multi-line report (the search table, the block of a seed) reads as
    one block of the package."""

    def format(self, record):
        return '\n'.join(_PREFIX + line for line in super().format(record).split('\n'))


def set_verbose(on: bool = True) -> None:
    """Show (``on=True``) or silence (``on=False``) the reports and diagnostic messages of echostatenetwork.

    On: the logger ``'echostatenetwork'`` takes the level DEBUG and one stream handler on stderr, which prefixes
    every line with ``[echostatenetwork]``; calling it again adds no second handler. Off: the handler is removed and
    the level returns to the default (NOTSET), so that the package is silent again unless the application's own
    logging configuration shows it.
    """
    logger = logging.getLogger('echostatenetwork')
    handlers = [h for h in logger.handlers if h.get_name() == _HANDLER]
    if on:
        if not handlers:
            handler = logging.StreamHandler()
            handler.set_name(_HANDLER)
            handler.setFormatter(_PrefixFormatter('%(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        for handler in handlers:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
