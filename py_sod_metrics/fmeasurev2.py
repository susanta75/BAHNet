import abc
import numpy as np
from .utils import TYPE, get_adaptive_threshold, prepare_data


class _BaseHandler:
    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
    ):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None
        self.sample_based = sample_based
        if with_binary:
            if self.sample_based:
                self.binary_results = []
            else:
                self.binary_results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        else:
            self.binary_results = None

    @abc.abstractmethod
    def __call__(self, *args, **kwds):
        pass

    @staticmethod
    def divide(numerator, denominator):
        denominator = np.array(denominator, dtype=TYPE)
        np.divide(numerator, denominator, out=denominator, where=denominator != 0)
        return denominator


class IOUHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(tp, tp + fp + fn)


class SpecificityHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(tn, tn + fp)


class DICEHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(2 * tp, tp + fn + tp + fp)


class OverallAccuracyHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(tp + tn, tp + fp + tn + fn)


class KappaHandler(_BaseHandler):

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        beta: float = 0.3,
    ):
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
        )

        self.beta = beta
        self.oa = OverallAccuracyHandler(False, False)

    def __call__(self, tp, fp, tn, fn):
        oa = self.oa(tp, fp, tn, fn)
        hpy_p = self.divide(
            (tp + fp) * (tp + fn) + (tn + fn) * (tn + tp),
            (tp + fp + tn + fn) ** 2,
        )
        return self.divide(oa - hpy_p, 1 - hpy_p)


class PrecisionHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(tp, tp + fp)


class RecallHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        return self.divide(tp, tp + fn)


class BERHandler(_BaseHandler):

    def __call__(self, tp, fp, tn, fn):
        fg = np.asarray(tp + fn, dtype=TYPE)
        bg = np.asarray(tn + fp, dtype=TYPE)
        np.divide(tp, fg, out=fg, where=fg != 0)
        np.divide(tn, bg, out=bg, where=bg != 0)
        return 1 - 0.5 * (fg + bg)


class FmeasureHandler(_BaseHandler):

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        beta: float = 0.3,
    ):
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
        )

        self.beta = beta
        self.precision = PrecisionHandler(False, False)
        self.recall = RecallHandler(False, False)

    def __call__(self, tp, fp, tn, fn):

        p = self.precision(tp, fp, tn, fn)
        r = self.recall(tp, fp, tn, fn)
        return self.divide((self.beta + 1) * p * r, self.beta * p + r)


class FmeasureV2:
    def __init__(self, metric_handlers: dict = None):
        self._metric_handlers = metric_handlers if metric_handlers else {}

    def add_handler(self, handler_name, metric_handler):
        self._metric_handlers[handler_name] = metric_handler

    @staticmethod
    def get_statistics(binary: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        TP = np.count_nonzero(binary[gt])
        FP = np.count_nonzero(binary[~gt])
        FN = FG - TP
        TN = BG - FP
        return {"tp": TP, "fp": FP, "tn": TN, "fn": FN}

    def adaptively_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary = pred >= adaptive_threshold
        return self.get_statistics(binary, gt, FG, BG)

    def dynamically_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        pred: np.ndarray = (pred * 255).astype(np.uint8)
        bins: np.ndarray = np.linspace(0, 256, 257)
        tp_hist, _ = np.histogram(pred[gt], bins=bins)
        fp_hist, _ = np.histogram(pred[~gt], bins=bins)

        tp_w_thrs = np.cumsum(np.flip(tp_hist))
        fp_w_thrs = np.cumsum(np.flip(fp_hist))

        TPs = tp_w_thrs
        FPs = fp_w_thrs
        FNs = FG - TPs
        TNs = BG - FPs
        return {"tp": TPs, "fp": FPs, "tn": TNs, "fn": FNs}

    def step(self, pred: np.ndarray, gt: np.ndarray):
        if not self._metric_handlers:
            raise ValueError("Please add your metric handler before using `step()`.")

        pred, gt = prepare_data(pred, gt)

        FG = np.count_nonzero(gt)
        BG = gt.size - FG

        dynamical_tpfptnfn = None
        adaptive_tpfptnfn = None
        binary_tpfptnfn = None
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None:
                if dynamical_tpfptnfn is None:
                    dynamical_tpfptnfn = self.dynamically_binarizing(
                        pred=pred, gt=gt, FG=FG, BG=BG
                    )
                handler.dynamic_results.append(handler(**dynamical_tpfptnfn))

            if handler.adaptive_results is not None:
                if adaptive_tpfptnfn is None:
                    adaptive_tpfptnfn = self.adaptively_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
                handler.adaptive_results.append(handler(**adaptive_tpfptnfn))

            if handler.binary_results is not None:
                if binary_tpfptnfn is None:
                    binary_tpfptnfn = self.get_statistics(binary=pred > 0.5, gt=gt, FG=FG, BG=BG)
                if handler.sample_based:
                    handler.binary_results.append(handler(**binary_tpfptnfn))
                else:
                    handler.binary_results["tp"] += binary_tpfptnfn["tp"]
                    handler.binary_results["fp"] += binary_tpfptnfn["fp"]
                    handler.binary_results["tn"] += binary_tpfptnfn["tn"]
                    handler.binary_results["fn"] += binary_tpfptnfn["fn"]

    def get_results(self) -> dict:
        results = {}
        for handler_name, handler in self._metric_handlers.items():
            res = {}
            if handler.dynamic_results is not None:
                res["dynamic"] = np.mean(np.array(handler.dynamic_results, dtype=TYPE), axis=0)
            if handler.adaptive_results is not None:
                res["adaptive"] = np.mean(np.array(handler.adaptive_results, dtype=TYPE))
            if handler.binary_results is not None:
                if handler.sample_based:
                    res["binary"] = np.mean(np.array(handler.binary_results, dtype=TYPE))
                else:
                    res["binary"] = np.mean(handler(**handler.binary_results))
            results[handler_name] = res
        return results
