import time
from dataclasses import dataclass

import numpy as np
from nltk.metrics import ghd as ghd
from nltk.metrics import pk as pk
from nltk.metrics import windowdiff as wd

# NOTE: only set k window size if segment lengths are highly variable


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """Segmentation evluation result container.

    Attributes:
        hyp_str: Hypothesized segmentation string representation.
        ref_str: Reference (ground truth) segmentation string representation.
        pk: Pk error metrics under different settings.
        wd: WindowDiff error metrics under different settings.
        ghd: Generalized Hamming Distance error metric
        runtime: Evaluation process execution time in seconds.
    """

    hyp_str: str
    ref_str: str
    pk: dict[str, float]
    wd: dict[str, float]
    ghd: float
    runtime: float

    def __str__(self) -> str:
        """Evaluation result container summary tostring function.

        Returns:
            str: Multi-line human-readable and interpretable summary of evaluation result.
        """
        width = 34

        lines = [
            "EvaluationResult",
            "-" * width,
            f"{'Metric':<20}{'Value':>14}",
            "-" * width,
            f"{'Pk (small)':<20}{self.pk['small']:>14.4f}",
            f"{'Pk (default)':<20}{self.pk['default']:>14.4f}",
            f"{'Pk (large)':<20}{self.pk['large']:>14.4f}",
            f"{'Pk (nltk)':<20}{self.pk['nltk']:>14.4f}",
            "",
            f"{'WD (small)':<20}{self.wd['small']:>14.4f}",
            f"{'WD (default)':<20}{self.wd['default']:>14.4f}",
            f"{'WD (large)':<20}{self.wd['large']:>14.4f}",
            "",
            f"{'GHD':<20}{self.ghd:>14.4f}",
            f"{'Runtime (s)':<20}{self.runtime:>14.4f}",
            "-" * width,
            f"REF {self.ref_str}",
            f"HYP {self.hyp_str}",
        ]
        return "\n".join(lines)


class SegmentationEvaluator:
    """Stateful evaluator for comparing segmentation results against ground truth.

    Attributes:
        ins_cost: Cost of inserting a boundary.
        del_cost: Cost of deleting a boundary.
        shift_cost_coeff: Cost of shifting a boundary.
        boundary_symbol: Character symbol that denotes a boundary.
    """

    def __init__(
        self,
        ins_cost=2.0,
        del_cost=2.0,
        shift_cost_coeff=1.0,
        boundary_symbol="1",
    ):

        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.shift_cost_coeff = shift_cost_coeff
        self.boundary_symbol = boundary_symbol

    def evaluate(
        self,
        ref_len: list[int],
        hyp_len: list[int],
    ) -> EvaluationResult:
        """Evaluation function

        Args:
            ref_len: Reference (ground truth) segment lengths.
            hyp_len: Hypothesized segment lengths.

        Returns:
            EvaluationResult: Evaluation result container.
        """
        start_time = time.perf_counter()

        self._validate_input(ref_len, hyp_len)

        # string boundary representation
        ref_str = self._lengths_to_str(ref_len)
        hyp_str = self._lengths_to_str(hyp_len)

        self._validate_conversion(str_rep=ref_str, len_rep=ref_len)
        self._validate_conversion(str_rep=hyp_str, len_rep=hyp_len)

        ghd_score = ghd(
            ref_str,
            hyp_str,
            self.ins_cost,
            self.del_cost,
            self.shift_cost_coeff,
            boundary=self.boundary_symbol,
        )

        pk_scores = {}
        wd_scores = {}

        k_values = self._k_values(ref_str, ref_len)
        for k_name, k in k_values.items():
            pk_scores[k_name] = pk(ref_str, hyp_str, k=k, boundary=self.boundary_symbol)
            wd_scores[k_name] = wd(ref_str, hyp_str, k=k, boundary=self.boundary_symbol)

        # optional: include nltk default (k=None)
        pk_scores["nltk"] = pk(ref_str, hyp_str, k=None, boundary=self.boundary_symbol)

        end_time = time.perf_counter()
        runtime = end_time - start_time

        return EvaluationResult(
            hyp_str=hyp_str,
            ref_str=ref_str,
            pk=pk_scores,
            wd=wd_scores,
            ghd=ghd_score,
            runtime=runtime,
        )

    def _lengths_to_str(self, lengths: list[int]) -> str:
        s = []
        for i, length in enumerate(lengths):
            s.extend(["0"] * (length - 1))
            if i != len(lengths) - 1:
                s.append("1")
        return "".join(s)

    def _k_values(self, ref_str, ref_lengths: list[int]) -> dict[str, int]:

        # percentile approach
        k_small = max(1, int(round(np.percentile(ref_lengths, 25) / 2)))
        # k_default = max(1, int(round(np.mean(ref_lengths) / 2)))
        k_default = self._nltk_default_k(ref_str)
        k_large = max(1, int(round(np.percentile(ref_lengths, 75) / 2)))

        # min and max approach
        # k_small = max(1, int(round(min(ref_lengths) / 2)))
        # k_default = int(round((sum(ref_lengths) / len(ref_lenghts)) / 2))
        # k_large = int(round(max(ref_lengths) / 2))

        return {
            "small": k_small,
            "default": k_default,
            "large": k_large,
        }

    def _validate_input(self, ref_len, hyp_len):
        # TODO: add more meaningful error messages
        ref_sum = sum(ref_len)
        hyp_sum = sum(hyp_len)

        if ref_sum != hyp_sum:
            raise ValueError(
                f"Segmentations must cover the same total length "
                f"(number of sentences).\n"
                f"Got: sum(ref_len)={ref_sum}, sum(hyp_len)={hyp_sum}"
            )

        invalid_ref = [len for len in ref_len if len <= 0]
        invalid_hyp = [len for len in hyp_len if len <= 0]

        if invalid_ref or invalid_hyp:
            raise ValueError(
                "Segment lengths must all be positive integers.\n"
                f"Invalid values in ref_len: {invalid_ref}\n"
                f"Invalid values in hyp_len: {invalid_hyp}"
            )

    # accept len and str and test if it makes sense
    def _validate_conversion(
        self,
        str_rep: str,
        len_rep: list[int],
    ):
        assert len(str_rep) == sum(len_rep) - 1

    def _nltk_default_k(self, ref_str, boundary: str = "1") -> int:
        n = len(ref_str)
        b = ref_str.count(boundary)

        if b == 0:
            raise ValueError("Reference contains no boundaries")

        return int(round(n / (2.0 * b)))
