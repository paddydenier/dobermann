import numpy as np
from nltk.metrics import pk as pk
from nltk.metrics import windowdiff as wd

# NOTE: only set k window size if segment lengths are highly variable

# NOTE: clearly mark which functions are symmetrics and which not


# TODO: Outsource evaluation to NLTK


# FIX: if data already lives in object dont pass it again
class SegmentationEvaluator:
    def __init__(self, ref_len: list[int], hyp_len: list[int]):

        # raw segment lengths
        self.ref_len = ref_len
        self.hyp_len = hyp_len

        # _validate_input
        self._validate_input()

        # string boundary representation
        self.ref_str = self._lengths_to_str(self.ref_len)
        self.hyp_str = self._lengths_to_str(self.hyp_len)

        # validate conversion consistency
        self._validate_conversion()

        self.k_values = self._k_values(ref_len)

        self.pkn = pk(self.ref_str, self.hyp_str, k=None, boundary="1")
        self.pkd = pk(
            self.ref_str, self.hyp_str, k=self.k_values["default"], boundary="1"
        )

        self.metrics = self.evaluate()

    from nltk.metrics import pk
    from nltk.metrics import windowdiff as wd

    def evaluate(self) -> dict[str, dict[str, float]]:
        results = {
            "pk": {},
            "wd": {},
        }

        for k_name, k in self.k_values.items():
            results["pk"][k_name] = pk(self.ref_str, self.hyp_str, k=k, boundary="1")
            results["wd"][k_name] = wd(self.ref_str, self.hyp_str, k=k, boundary="1")

        # optional: include nltk default (k=None)
        results["pk"]["nltk"] = pk(self.ref_str, self.hyp_str, k=None, boundary="1")

        return results

        # for each k evaluate pk and wd score.
        # self.pk = pk(self.ref_lengths, self.hyp_lenghts, k=None, boundary="1")
        # self.wd = wd(self.ref_lengths, self.hyp_lenghts, k=self.k, boundary="1")

    def _lengths_to_str(self, lengths: list[int]) -> str:
        s = []
        for i, length in enumerate(lengths):
            s.extend(["0"] * (length - 1))
            if i != len(lengths) - 1:
                s.append("1")
        return "".join(s)

    def _k_values(self, ref_lengths: list[int]) -> dict[str, int]:

        # percentile approach
        k_small = max(1, int(round(np.percentile(ref_lengths, 25) / 2)))
        # k_default = max(1, int(round(np.mean(ref_lengths) / 2)))
        k_default = self._nltk_default_k()
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

    def _validate_input(self):
        # TODO: add more meaningful error messages
        ref_sum = sum(self.ref_len)
        hyp_sum = sum(self.hyp_len)

        if ref_sum != hyp_sum:
            raise ValueError(
                f"Segmentations must cover the same total length "
                f"(number of sentences).\n"
                f"Got: sum(ref_len)={ref_sum}, sum(hyp_len)={hyp_sum}"
            )

        invalid_ref = [len for len in self.ref_len if len <= 0]
        invalid_hyp = [len for len in self.hyp_len if len <= 0]

        if invalid_ref or invalid_hyp:
            raise ValueError(
                "Segment lengths must all be positive integers.\n"
                f"Invalid values in ref_len: {invalid_ref}\n"
                f"Invalid values in hyp_len: {invalid_hyp}"
            )

    def _validate_conversion(self):
        assert len(self.ref_str) == sum(self.ref_len) - 1
        assert len(self.hyp_str) == sum(self.hyp_len) - 1

    def _nltk_default_k(self, boundary: str = "1") -> int:
        n = len(self.ref_str)
        b = self.ref_str.count(boundary)

        if b == 0:
            raise ValueError("Reference contains no boundaries")

        return int(round(n / (2.0 * b)))
