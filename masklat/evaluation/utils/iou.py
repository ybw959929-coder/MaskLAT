import numpy as np
from tabulate import tabulate


class IouStat:
    """Accumulate per-class RefSeg IoU statistics.

    Public ``cIoU`` and ``gIoU`` values use the conventional RefSeg percentage
    scale and therefore lie in ``[0, 100]``.
    """

    def __init__(self, cat_names=("ignore", "refer")):
        self.cat_names = tuple(cat_names)
        self.num_cats = len(self.cat_names)
        if self.num_cats <= 0:
            raise ValueError("cat_names must contain at least one category")
        self.reset()

    def update(self, intersection, union, n=1):
        """
        Args:
            intersection: array-like, shape (num_cats,)
            union: array-like, shape (num_cats,)
            n: number of samples
        """
        intersection = np.asarray(intersection, dtype=np.float64)
        union = np.asarray(union, dtype=np.float64)
        expected_shape = (self.num_cats,)
        if intersection.shape != expected_shape or union.shape != expected_shape:
            raise ValueError(
                "intersection and union must each have shape "
                f"{expected_shape}, got {intersection.shape} and {union.shape}"
            )
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        if not np.isfinite(intersection).all() or not np.isfinite(union).all():
            raise ValueError("intersection and union must contain only finite values")
        if (intersection < 0).any() or (union < 0).any():
            raise ValueError("intersection and union must be non-negative")
        if (intersection > union).any():
            raise ValueError("intersection cannot exceed union")

        self.intersection += intersection
        self.union += union
        self.count += n

        iou_per_sample = np.ones(self.num_cats, dtype=np.float64)
        np.divide(
            intersection,
            union,
            out=iou_per_sample,
            where=union > 0,
        )
        self.acc_iou += iou_per_sample * n

    def average(self):
        # cIoU (cumulative IoU)
        self.ciou.fill(100.0)
        np.divide(
            self.intersection * 100.0,
            self.union,
            out=self.ciou,
            where=self.union > 0,
        )

        # gIoU (global mean IoU)
        if self.count > 0:
            self.giou[:] = self.acc_iou / self.count * 100.0
        else:
            self.giou.fill(0.0)

        for metric_name, values in (("cIoU", self.ciou), ("gIoU", self.giou)):
            if not np.isfinite(values).all():
                raise ValueError(f"{metric_name} contains a non-finite value")
            if (values < -1e-12).any() or (values > 100.0 + 1e-12).any():
                raise ValueError(
                    f"{metric_name} escaped the percentage range [0, 100]: "
                    f"{values.tolist()}"
                )

    def reset(self):
        self.intersection = np.zeros(self.num_cats, dtype=np.float64)
        self.union = np.zeros(self.num_cats, dtype=np.float64)
        self.count = 0
        self.acc_iou = np.zeros(self.num_cats, dtype=np.float64)
        self.ciou = np.zeros(self.num_cats, dtype=np.float64)
        self.giou = np.zeros(self.num_cats, dtype=np.float64)

    def __repr__(self) -> str:
        headers = ["", "cIoU", "gIoU"]
        data = []
        for i, cat_name in enumerate(self.cat_names):
            data.append(
                [
                    cat_name,
                    self.ciou[i],
                    self.giou[i],
                ]
            )

        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        return str(table)
