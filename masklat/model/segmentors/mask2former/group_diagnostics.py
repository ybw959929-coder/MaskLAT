"""Distributed-safe aggregation and formatting for topology-group diagnostics.

This module deliberately has no logging side effects.  Callers may collect
statistics while running a model, globally reduce them after evaluation, and
print the returned tables on rank zero.
"""

from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor


Number = Union[int, float, Tensor]

REFSEG_VARIANTS = (
    "selected_query_mask",
    "selected_group_top_class_query_mask",
    "selected_group_quality_query_mask",
    "selected_group_oracle_member_mask",
    "global_group_oracle_member_mask",
    "global_200q_oracle_query_mask",
)

GT_SELECT_VARIANTS = frozenset(
    {
        "selected_group_oracle_member_mask",
        "global_group_oracle_member_mask",
        "global_200q_oracle_query_mask",
    }
)

REFSEG_COLUMNS = (
    "variant",
    "cIoU",
    "gIoU",
    "dC",
    "dG",
    "improve%",
    "worsen%",
    "GT-select",
)
GROUP_STRUCTURE_COLUMNS = (
    "stage",
    "avg_groups",
    "avg_group_size",
    "singleton%",
    "largest_group",
    "group_purity",
    "good_group_count7",
    "duplicate_group_count7",
    "bridge%",
)
ROUTING_MEMBER_COLUMNS = (
    "selected_cond_probability",
    "selected_cond_margin",
    "selected_cond_is_gt%",
    "selected_cond_is_bg%",
    "selected_cond_is_other%",
    "selected_group_contains_good7%",
    "selected_group_same_as_oracle_group%",
    "success7%",
    "wrong_group7%",
    "wrong_member7%",
    "no_good_query7%",
    "first_good_group_rank_mean",
    "first_good_group_rank_median",
    "first_good_group_rank_top1%",
    "first_good_group_rank_top3%",
    "first_good_group_rank_top5%",
    "within_group_member_gap_gIoU",
    "cross_group_routing_gap_gIoU",
    "varAvail7%",
    "varSelected7%",
    "rescueOpportunity7%",
    "rescue7%",
    "matLost7%",
    "goodConcentration7",
    "bestGroupPurity7",
    "condIoUSpearman",
    "condIoUSpearmanN",
)
MULTIGT_COLUMNS = (
    "task",
    "targets",
    "conditions",
    "groups",
    "matched_groups",
    "bg_groups",
    "same_cond_multi_gt_samples%",
    "unmatched_targets",
    "group_recall50",
    "group_recall70",
    "duplicate_prediction_rate",
)
STAGE_COLUMNS = (
    "stage",
    "feedback_delta_norm",
    "query_norm",
    "group_feature_norm",
    "alpha_region",
    "alpha_fb",
    "loss_group_cls",
    "loss_group_mask",
    "loss_group_dice",
)
STAGEWISE_INTERNAL_COLUMNS = (
    "stage",
    "QoracleG",
    "noGood7%",
    "G@1_70%",
    "routeDeltaG",
    "condIoUrho",
    "rhoN",
    "goodConc7",
    "bestPur7",
    "wrongM7%",
    "memberDeltaG",
)


def _key(*parts: object) -> str:
    return "\x1f".join(str(part) for part in parts)


def _finite_values(value: Number, name: str) -> Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        return tensor.cpu()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor.cpu()


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _format_number(
    value: Optional[float],
    *,
    digits: int = 4,
    integer: bool = False,
    percent: bool = False,
) -> str:
    if value is None:
        return "N/A"
    if percent:
        value *= 100.0
    if integer:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


class GroupDiagnostics:
    """Accumulate group diagnostics without printing from model ``forward``.

    Statistics are stored as additive sufficient statistics so DDP aggregation
    uses SUM, except ``largest_group`` which uses MAX.  Metrics requiring GT are
    optional; if they are never supplied their terminal cells are rendered as
    ``N/A`` rather than inferred from unrelated forward tensors.
    """

    def __init__(self, num_stages: int = 10):
        if int(num_stages) <= 0:
            raise ValueError("num_stages must be positive")
        self.num_stages = int(num_stages)
        self.reset()

    def reset(self) -> None:
        self._totals: Dict[str, float] = {}
        self._mean_sums: Dict[str, float] = {}
        self._mean_counts: Dict[str, float] = {}
        self._maxima: Dict[str, float] = {}
        self._variant_gt_select = {
            variant: variant in GT_SELECT_VARIANTS for variant in REFSEG_VARIANTS
        }
        self._globally_reduced = False

    def _ensure_mutable(self) -> None:
        if self._globally_reduced:
            raise RuntimeError("cannot update GroupDiagnostics after global reduction")

    def _add_total(self, key: str, value: Number, name: str) -> None:
        values = _finite_values(value, name)
        if values.numel():
            self._totals[key] = self._totals.get(key, 0.0) + float(values.sum().item())

    def _observe(self, key: str, value: Optional[Number], name: str) -> None:
        if value is None:
            return
        values = _finite_values(value, name)
        if not values.numel():
            return
        self._mean_sums[key] = self._mean_sums.get(key, 0.0) + float(values.sum().item())
        self._mean_counts[key] = self._mean_counts.get(key, 0.0) + float(values.numel())

    def _observe_weighted(
        self,
        key: str,
        value: Optional[Number],
        weight: Number,
        name: str,
    ) -> None:
        if value is None:
            return
        values = _finite_values(value, name)
        weights = _finite_values(weight, f"{name} weight")
        if weights.numel() == 1:
            weights = weights.expand_as(values)
        if values.shape != weights.shape:
            raise ValueError(f"{name} and its weights must be broadcastable scalars or match")
        if bool((weights < 0).any()):
            raise ValueError(f"{name} weights must be non-negative")
        total_weight = float(weights.sum().item())
        if total_weight <= 0:
            return
        self._mean_sums[key] = self._mean_sums.get(key, 0.0) + float(
            (values * weights).sum().item()
        )
        self._mean_counts[key] = self._mean_counts.get(key, 0.0) + total_weight

    def _observe_indicators(self, key: str, value: Optional[Number], name: str) -> None:
        if value is None:
            return
        values = _finite_values(value, name)
        if bool(((values < 0) | (values > 1)).any()):
            raise ValueError(f"{name} indicators must be in [0,1]")
        self._observe(key, values, name)

    def _set_max(self, key: str, value: Number, name: str) -> None:
        values = _finite_values(value, name)
        if values.numel():
            current = self._maxima.get(key, float("-inf"))
            self._maxima[key] = max(current, float(values.max().item()))

    def _mean(self, key: str) -> Optional[float]:
        count = self._mean_counts.get(key, 0.0)
        if count <= 0:
            return None
        return self._mean_sums.get(key, 0.0) / count

    def _ratio(self, numerator_key: str, denominator_key: str) -> Optional[float]:
        denominator = self._totals.get(denominator_key, 0.0)
        if denominator <= 0:
            return None
        return self._totals.get(numerator_key, 0.0) / denominator

    @torch.no_grad()
    def update_stage(
        self,
        stage_output,
        *,
        stage_index: Optional[int] = None,
        group_purity: Optional[Number] = None,
        good_group_count7: Optional[Number] = None,
        duplicate_group_count7: Optional[Number] = None,
        bridge_flags: Optional[Number] = None,
        alpha_region: Optional[Number] = None,
        alpha_fb: Optional[Number] = None,
        losses: Optional[Mapping[str, Number]] = None,
    ) -> None:
        """Collect one stage's structural and optimization statistics.

        ``stage_output`` is duck-typed to avoid coupling this utility to the
        decoder dataclass.  Forward-only tensors provide group counts/sizes and
        norms.  GT-dependent arguments are optional and remain ``N/A`` when
        omitted.
        """

        self._ensure_mutable()
        if stage_index is None:
            stage_index = int(stage_output.stage_index)
        stage_index = int(stage_index)
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(f"stage {stage_index} outside [0,{self.num_stages})")

        valid = torch.as_tensor(stage_output.group_valid_mask).detach().to(torch.bool)
        sizes = torch.as_tensor(stage_output.group_sizes).detach()
        if valid.ndim != 2 or sizes.shape != valid.shape:
            raise ValueError("group_valid_mask and group_sizes must have matching [B,Q] shapes")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid group")
        if bool(sizes[~valid].ne(0).any()):
            raise ValueError("non-root rows of group_sizes must be zero")
        valid_sizes = sizes[valid].to(torch.float64)
        if not bool(torch.isfinite(valid_sizes).all()) or bool((valid_sizes <= 0).any()):
            raise ValueError("valid group sizes must be finite and positive")
        per_sample_members = sizes.masked_fill(~valid, 0).sum(dim=1).to(torch.float64)
        expected_members = torch.full_like(per_sample_members, valid.shape[1])
        if not bool(torch.equal(per_sample_members, expected_members)):
            raise ValueError("valid group sizes must partition all Q queries in every sample")
        query_to_group = getattr(stage_output, "query_to_group", None)
        if query_to_group is not None:
            query_to_group = torch.as_tensor(query_to_group).detach()
            if query_to_group.shape != valid.shape:
                raise ValueError("query_to_group must have shape [B,Q]")
            query_ids = torch.arange(valid.shape[1], device=query_to_group.device)
            physical_roots = query_to_group.eq(query_ids.unsqueeze(0))
            if not bool(torch.equal(physical_roots.cpu(), valid.cpu())):
                raise ValueError("group_valid_mask must be true exactly at physical root rows")

        prefix = _key("structure", stage_index)
        self._add_total(_key(prefix, "samples"), valid.shape[0], "sample count")
        self._add_total(_key(prefix, "groups"), valid.sum(), "group count")
        self._add_total(_key(prefix, "members"), valid_sizes.sum(), "group members")
        self._add_total(
            _key(prefix, "singletons"),
            valid_sizes.eq(1).sum(),
            "singleton count",
        )
        self._set_max(_key(prefix, "largest"), valid_sizes, "group sizes")

        if group_purity is not None:
            purity = torch.as_tensor(group_purity).detach()
            if purity.shape == valid.shape:
                purity = purity[valid]
            self._observe(_key(prefix, "purity"), purity, "group_purity")
        self._observe(
            _key(prefix, "good_group_count7"),
            good_group_count7,
            "good_group_count7",
        )
        self._observe(
            _key(prefix, "duplicate_group_count7"),
            duplicate_group_count7,
            "duplicate_group_count7",
        )
        if bridge_flags is not None:
            flags = torch.as_tensor(bridge_flags).detach()
            if flags.shape == valid.shape:
                eligible = valid & sizes.gt(1)
                flags = flags[eligible]
            self._observe_indicators(_key(prefix, "bridge"), flags, "bridge_flags")

        stage_prefix = _key("stage", stage_index)
        query_states = getattr(stage_output, "query_states", None)
        if query_states is not None:
            query_norms = torch.as_tensor(query_states).detach().to(torch.float32).norm(dim=-1)
            self._observe(_key(stage_prefix, "query_norm"), query_norms, "query_norm")
        group_features = getattr(stage_output, "group_features", None)
        if group_features is not None:
            group_features = torch.as_tensor(group_features).detach()
            if group_features.shape[:2] != valid.shape:
                raise ValueError("group_features must have the same [B,Q] prefix as group_valid_mask")
            group_norms = group_features.to(torch.float32).norm(dim=-1)[valid]
            self._observe(
                _key(stage_prefix, "group_feature_norm"),
                group_norms,
                "group_feature_norm",
            )
        feedback_delta_norm = getattr(stage_output, "feedback_delta_norm", None)
        if feedback_delta_norm is not None and torch.as_tensor(feedback_delta_norm).numel() == 1:
            self._observe_weighted(
                _key(stage_prefix, "feedback_delta_norm"),
                feedback_delta_norm,
                valid.shape[0],
                "feedback_delta_norm",
            )
        else:
            self._observe(
                _key(stage_prefix, "feedback_delta_norm"),
                feedback_delta_norm,
                "feedback_delta_norm",
            )
        self._observe(_key(stage_prefix, "alpha_region"), alpha_region, "alpha_region")
        self._observe(_key(stage_prefix, "alpha_fb"), alpha_fb, "alpha_fb")

        losses = losses or {}
        aliases = {
            "loss_group_cls": ("loss_group_cls", "group_cls"),
            "loss_group_mask": ("loss_group_mask", "group_mask"),
            "loss_group_dice": ("loss_group_dice", "group_dice"),
        }
        for output_name, candidates in aliases.items():
            value = next((losses[name] for name in candidates if name in losses), None)
            if value is not None and torch.as_tensor(value).numel() == 1:
                self._observe_weighted(
                    _key(stage_prefix, output_name),
                    value,
                    valid.shape[0],
                    output_name,
                )
            else:
                self._observe(_key(stage_prefix, output_name), value, output_name)

    def update_stages(self, stage_outputs: Sequence[object]) -> None:
        """Convenience wrapper for a complete st0--stN stage sequence."""

        for stage_output in stage_outputs:
            self.update_stage(stage_output)

    def update_stagewise_internal(
        self,
        stage_index: int,
        *,
        query_oracle_giou: Optional[Number] = None,
        no_good_query7: Optional[Number] = None,
        selected_group_contains_good7: Optional[Number] = None,
        routing_gap_giou: Optional[Number] = None,
        cond_iou_spearman: Optional[Number] = None,
        good_concentration7: Optional[Number] = None,
        best_group_purity7: Optional[Number] = None,
        wrong_member7: Optional[Number] = None,
        member_gap_giou: Optional[Number] = None,
    ) -> None:
        """Collect GT-backed stagewise internal/mechanism diagnostics.

        These trajectories describe associations inside the decoder; they do
        not by themselves establish a causal effect of feedback.  The member
        selector fields are intentionally optional because stages before the
        final stage do not have a formal learned quality representative.
        """

        self._ensure_mutable()
        stage_index = int(stage_index)
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(
                f"stage {stage_index} outside [0,{self.num_stages})"
            )
        prefix = _key("stage_internal", stage_index)
        self._observe(
            _key(prefix, "query_oracle_giou"),
            query_oracle_giou,
            "query_oracle_giou",
        )
        for name, value in (
            ("no_good_query7", no_good_query7),
            (
                "selected_group_contains_good7",
                selected_group_contains_good7,
            ),
            ("wrong_member7", wrong_member7),
        ):
            self._observe_indicators(_key(prefix, name), value, name)
        self._observe(
            _key(prefix, "routing_gap_giou"),
            routing_gap_giou,
            "routing_gap_giou",
        )
        self._observe(
            _key(prefix, "member_gap_giou"),
            member_gap_giou,
            "member_gap_giou",
        )
        for name, value in (
            ("good_concentration7", good_concentration7),
            ("best_group_purity7", best_group_purity7),
        ):
            if value is not None:
                values = _finite_values(value, name)
                if bool(((values < 0) | (values > 1)).any()):
                    raise ValueError(f"{name} must be in [0,1]")
            self._observe(_key(prefix, name), value, name)
        if cond_iou_spearman is not None:
            values = _finite_values(
                cond_iou_spearman,
                "cond_iou_spearman",
            )
            if bool(((values < -1) | (values > 1)).any()):
                raise ValueError("cond_iou_spearman must be in [-1,1]")
        self._observe(
            _key(prefix, "cond_iou_spearman"),
            cond_iou_spearman,
            "cond_iou_spearman",
        )

    def update_refseg_variant(
        self,
        variant: str,
        *,
        intersection: Number,
        union: Number,
        sample_iou: Optional[Number] = None,
        improved: Optional[Number] = None,
        worsened: Optional[Number] = None,
        gt_select: Optional[bool] = None,
    ) -> None:
        """Add RefSeg sufficient statistics for one formal prediction variant."""

        self._ensure_mutable()
        if not variant:
            raise ValueError("variant must be non-empty")
        intersections = _finite_values(intersection, "intersection")
        unions = _finite_values(union, "union")
        if bool((intersections < 0).any()) or bool((unions < 0).any()):
            raise ValueError("intersection and union must be non-negative")
        prefix = _key("refseg", variant)
        self._add_total(_key(prefix, "intersection"), intersections, "intersection")
        self._add_total(_key(prefix, "union"), unions, "union")

        if sample_iou is None and intersections.numel() > 1 and intersections.shape == unions.shape:
            valid_union = unions > 0
            sample_iou = intersections[valid_union] / unions[valid_union]
        self._observe(_key(prefix, "giou"), sample_iou, "sample_iou")
        self._observe_indicators(_key(prefix, "improved"), improved, "improved")
        self._observe_indicators(_key(prefix, "worsened"), worsened, "worsened")
        if gt_select is not None:
            self._variant_gt_select[variant] = bool(gt_select)
        elif variant not in self._variant_gt_select:
            self._variant_gt_select[variant] = variant in GT_SELECT_VARIANTS

    def update_routing(
        self,
        *,
        selected_cond_probability: Optional[Number] = None,
        selected_cond_margin: Optional[Number] = None,
        selected_cond_is_gt: Optional[Number] = None,
        selected_cond_is_bg: Optional[Number] = None,
        selected_cond_is_other: Optional[Number] = None,
        selected_group_contains_good7: Optional[Number] = None,
        selected_group_same_as_oracle_group: Optional[Number] = None,
        success7: Optional[Number] = None,
        wrong_group7: Optional[Number] = None,
        wrong_member7: Optional[Number] = None,
        no_good_query7: Optional[Number] = None,
        first_good_group_rank: Optional[Number] = None,
        within_group_member_gap_giou: Optional[Number] = None,
        cross_group_routing_gap_giou: Optional[Number] = None,
    ) -> None:
        """Add GT-dependent routing/member observations.

        Indicator inputs are booleans or values in ``[0,1]``.  First-good
        ranks are one-based.  Gap values are differences of per-sample IoU;
        their dataset means therefore decompose gIoU (mean IoU), not cIoU.
        """

        self._ensure_mutable()
        self._observe(
            _key("routing", "selected_cond_probability"),
            selected_cond_probability,
            "selected_cond_probability",
        )
        self._observe(
            _key("routing", "selected_cond_margin"),
            selected_cond_margin,
            "selected_cond_margin",
        )
        indicators = {
            "selected_cond_is_gt": selected_cond_is_gt,
            "selected_cond_is_bg": selected_cond_is_bg,
            "selected_cond_is_other": selected_cond_is_other,
            "selected_group_contains_good7": selected_group_contains_good7,
            "selected_group_same_as_oracle_group": selected_group_same_as_oracle_group,
            "success7": success7,
            "wrong_group7": wrong_group7,
            "wrong_member7": wrong_member7,
            "no_good_query7": no_good_query7,
        }
        for name, value in indicators.items():
            self._observe_indicators(_key("routing", name), value, name)

        if first_good_group_rank is not None:
            ranks = _finite_values(first_good_group_rank, "first_good_group_rank")
            if bool((ranks < 1).any()) or bool(ranks.ne(ranks.round()).any()):
                raise ValueError("first_good_group_rank must contain positive one-based integers")
            self._observe(_key("routing", "rank"), ranks, "first_good_group_rank")
            for rank in ranks.to(torch.int64).tolist():
                self._add_total(_key("routing", "rank_hist", rank), 1, "rank histogram")
        self._observe(
            _key("routing", "within_group_member_gap_giou"),
            within_group_member_gap_giou,
            "within_group_member_gap_giou",
        )
        self._observe(
            _key("routing", "cross_group_routing_gap_giou"),
            cross_group_routing_gap_giou,
            "cross_group_routing_gap_giou",
        )

    def update_idea_diagnostics(
        self,
        *,
        variant_available7: Optional[Number] = None,
        variant_selected7: Optional[Number] = None,
        rescue_opportunity7: Optional[Number] = None,
        rescue7: Optional[Number] = None,
        matched_good_lost7: Optional[Number] = None,
        good_concentration7: Optional[Number] = None,
        best_group_purity7: Optional[Number] = None,
        cond_iou_spearman: Optional[Number] = None,
    ) -> None:
        """Add final-stage diagnostics specific to the topology-group idea.

        ``variant_available7`` and ``variant_selected7`` are sample-level
        indicators.  A variant is an IoU>=0.7 Query other than the
        deterministic full-resolution evaluation matcher's selected member.
        That matcher reuses the configured training costs but is not a record
        of the stochastic point-sampled assignment used by a historical
        training batch.  ``rescue_opportunity7`` is observed over every
        sample.  ``rescue7`` is supplied only for opportunities, so its mean
        is the conditional rescue success rate.  Likewise,
        ``matched_good_lost7`` is supplied only when the evaluation-matched
        member is good, making its mean the conditional loss rate.

        Concentration and purity are observed only for samples containing at
        least one good Query.  Spearman is observed only when at least two
        Groups exist and neither the Cond scores nor Group oracle IoUs are
        constant.  Their ``_mean_counts`` values are the exact valid
        denominators exported by :meth:`to_dict`.
        """

        self._ensure_mutable()
        for name, value in (
            ("variant_available7", variant_available7),
            ("variant_selected7", variant_selected7),
            ("rescue_opportunity7", rescue_opportunity7),
            ("rescue7", rescue7),
            ("matched_good_lost7", matched_good_lost7),
        ):
            self._observe_indicators(_key("idea", name), value, name)
        for name, value in (
            ("good_concentration7", good_concentration7),
            ("best_group_purity7", best_group_purity7),
        ):
            if value is not None:
                values = _finite_values(value, name)
                if bool(((values < 0) | (values > 1)).any()):
                    raise ValueError(f"{name} must be in [0,1]")
            self._observe(_key("idea", name), value, name)
        if cond_iou_spearman is not None:
            values = _finite_values(cond_iou_spearman, "cond_iou_spearman")
            if bool(((values < -1) | (values > 1)).any()):
                raise ValueError("cond_iou_spearman must be in [-1,1]")
        self._observe(
            _key("idea", "cond_iou_spearman"),
            cond_iou_spearman,
            "cond_iou_spearman",
        )

    def update_multigt(
        self,
        task: str,
        *,
        targets: Optional[Number] = None,
        conditions: Optional[Number] = None,
        groups: Optional[Number] = None,
        matched_groups: Optional[Number] = None,
        bg_groups: Optional[Number] = None,
        same_cond_multi_gt_samples: Optional[Number] = None,
        sample_count: Optional[Number] = None,
        unmatched_targets: Optional[Number] = None,
        group_recall50_hits: Optional[Number] = None,
        group_recall70_hits: Optional[Number] = None,
        recall_denominator: Optional[Number] = None,
        duplicate_predictions: Optional[Number] = None,
        duplicate_denominator: Optional[Number] = None,
    ) -> None:
        """Add task-level multi-GT counts.

        Rate cells require explicit denominators, except recalls may reuse
        ``targets`` and duplicate rate may reuse ``groups`` from the same call.
        Missing inputs stay ``N/A``.
        """

        self._ensure_mutable()
        if not task:
            raise ValueError("task must be non-empty")
        prefix = _key("multigt", task)
        totals = {
            "targets": targets,
            "conditions": conditions,
            "groups": groups,
            "matched_groups": matched_groups,
            "bg_groups": bg_groups,
            "unmatched_targets": unmatched_targets,
        }
        for name, value in totals.items():
            if value is not None:
                values = _finite_values(value, name)
                if bool((values < 0).any()):
                    raise ValueError(f"{name} must be non-negative")
                self._add_total(_key(prefix, name), values, name)

        ratio_inputs = (
            (
                "same_cond_multi_gt_samples",
                same_cond_multi_gt_samples,
                sample_count,
            ),
            (
                "group_recall50",
                group_recall50_hits,
                (
                    recall_denominator
                    if recall_denominator is not None
                    else targets if group_recall50_hits is not None else None
                ),
            ),
            (
                "group_recall70",
                group_recall70_hits,
                (
                    recall_denominator
                    if recall_denominator is not None
                    else targets if group_recall70_hits is not None else None
                ),
            ),
            (
                "duplicate_prediction_rate",
                duplicate_predictions,
                (
                    duplicate_denominator
                    if duplicate_denominator is not None
                    else groups if duplicate_predictions is not None else None
                ),
            ),
        )
        for name, numerator, denominator in ratio_inputs:
            if numerator is None and denominator is None:
                continue
            if numerator is None or denominator is None:
                raise ValueError(f"{name} requires both numerator and denominator")
            numerator_values = _finite_values(numerator, f"{name} numerator")
            denominator_values = _finite_values(denominator, f"{name} denominator")
            if bool((numerator_values < 0).any()) or bool((denominator_values < 0).any()):
                raise ValueError(f"{name} counts must be non-negative")
            self._add_total(
                _key(prefix, name, "numerator"),
                numerator_values,
                f"{name} numerator",
            )
            self._add_total(
                _key(prefix, name, "denominator"),
                denominator_values,
                f"{name} denominator",
            )

    def merge_(self, other: "GroupDiagnostics") -> "GroupDiagnostics":
        """Merge another rank/local shard using the same sum/max semantics."""

        self._ensure_mutable()
        if self.num_stages != other.num_stages:
            raise ValueError("cannot merge diagnostics with different num_stages")
        for destination, source in (
            (self._totals, other._totals),
            (self._mean_sums, other._mean_sums),
            (self._mean_counts, other._mean_counts),
        ):
            for key, value in source.items():
                destination[key] = destination.get(key, 0.0) + value
        for key, value in other._maxima.items():
            self._maxima[key] = max(self._maxima.get(key, float("-inf")), value)
        self._variant_gt_select.update(other._variant_gt_select)
        return self

    @staticmethod
    def _collective_device(group=None, device=None) -> torch.device:
        if device is not None:
            return torch.device(device)
        backend = str(dist.get_backend(group)).lower()
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL diagnostics reduction requires a CUDA device")
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _global_keys(local_keys: Sequence[str], group=None) -> Sequence[str]:
        world_size = dist.get_world_size(group)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, tuple(local_keys), group=group)
        return sorted({key for keys in gathered for key in keys})

    def _all_reduce_sum_dict(self, values: Dict[str, float], group, device) -> None:
        keys = self._global_keys(sorted(values), group)
        if not keys:
            return
        tensor = torch.tensor(
            [values.get(key, 0.0) for key in keys],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        reduced = tensor.cpu().tolist()
        values.clear()
        values.update(zip(keys, reduced))

    def _all_reduce_max_dict(self, values: Dict[str, float], group, device) -> None:
        keys = self._global_keys(sorted(values), group)
        if not keys:
            return
        tensor = torch.tensor(
            [values.get(key, float("-inf")) for key in keys],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
        reduced = tensor.cpu().tolist()
        values.clear()
        values.update(
            (key, value) for key, value in zip(keys, reduced) if value != float("-inf")
        )

    def all_reduce_(self, group=None, device=None) -> "GroupDiagnostics":
        """Globally aggregate all sufficient statistics in place.

        This is an idempotent no-op when distributed execution is unavailable,
        uninitialized, or has world size one.
        """

        if self._globally_reduced:
            return self
        if not dist.is_available() or not dist.is_initialized():
            return self
        if dist.get_world_size(group) <= 1:
            return self
        collective_device = self._collective_device(group=group, device=device)
        self._all_reduce_sum_dict(self._totals, group, collective_device)
        self._all_reduce_sum_dict(self._mean_sums, group, collective_device)
        self._all_reduce_sum_dict(self._mean_counts, group, collective_device)
        self._all_reduce_max_dict(self._maxima, group, collective_device)
        self._globally_reduced = True
        return self

    def global_sum(self, group=None, device=None) -> "GroupDiagnostics":
        """Alias for :meth:`all_reduce_` emphasizing SUM aggregation."""

        return self.all_reduce_(group=group, device=device)

    def _variant_metrics(self, variant: str) -> Tuple[Optional[float], Optional[float]]:
        prefix = _key("refseg", variant)
        union = self._totals.get(_key(prefix, "union"), 0.0)
        ciou = (
            self._totals.get(_key(prefix, "intersection"), 0.0) / union
            if union > 0
            else None
        )
        return ciou, self._mean(_key(prefix, "giou"))

    def format_refseg_table(self) -> str:
        baseline_ciou, baseline_giou = self._variant_metrics("selected_query_mask")
        variants = list(REFSEG_VARIANTS)
        variants.extend(sorted(set(self._variant_gt_select) - set(variants)))
        rows = []
        for variant in variants:
            ciou, giou = self._variant_metrics(variant)
            dc = ciou - baseline_ciou if ciou is not None and baseline_ciou is not None else None
            dg = giou - baseline_giou if giou is not None and baseline_giou is not None else None
            prefix = _key("refseg", variant)
            rows.append(
                (
                    variant,
                    _format_number(ciou, percent=True),
                    _format_number(giou, percent=True),
                    _format_number(dc, percent=True),
                    _format_number(dg, percent=True),
                    _format_number(self._mean(_key(prefix, "improved")), digits=2, percent=True),
                    _format_number(self._mean(_key(prefix, "worsened")), digits=2, percent=True),
                    str(bool(self._variant_gt_select.get(variant, False))),
                )
            )
        return _markdown_table(REFSEG_COLUMNS, rows)

    def format_ref_main_table(self) -> str:
        """Alias using the formal table's short name."""

        return self.format_refseg_table()

    def format_group_structure_table(self) -> str:
        rows = []
        for stage in range(self.num_stages):
            prefix = _key("structure", stage)
            samples = self._totals.get(_key(prefix, "samples"), 0.0)
            groups = self._totals.get(_key(prefix, "groups"), 0.0)
            members = self._totals.get(_key(prefix, "members"), 0.0)
            rows.append(
                (
                    f"st{stage}",
                    _format_number(groups / samples if samples > 0 else None, digits=3),
                    _format_number(members / groups if groups > 0 else None, digits=3),
                    _format_number(
                        self._ratio(_key(prefix, "singletons"), _key(prefix, "groups")),
                        digits=2,
                        percent=True,
                    ),
                    _format_number(self._maxima.get(_key(prefix, "largest")), integer=True),
                    _format_number(
                        self._mean(_key(prefix, "purity")),
                        digits=2,
                        percent=True,
                    ),
                    _format_number(self._mean(_key(prefix, "good_group_count7")), digits=3),
                    _format_number(
                        self._mean(_key(prefix, "duplicate_group_count7")),
                        digits=3,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "bridge")),
                        digits=2,
                        percent=True,
                    ),
                )
            )
        return _markdown_table(GROUP_STRUCTURE_COLUMNS, rows)

    def _rank_median(self) -> Optional[float]:
        histogram = []
        prefix = _key("routing", "rank_hist") + "\x1f"
        for key, count in self._totals.items():
            if key.startswith(prefix) and count > 0:
                histogram.append((int(key.rsplit("\x1f", 1)[-1]), count))
        if not histogram:
            return None
        histogram.sort()
        total = int(round(sum(count for _, count in histogram)))
        lower_position = (total + 1) // 2
        upper_position = (total + 2) // 2

        def value_at(position: int) -> float:
            cumulative = 0.0
            for rank, count in histogram:
                cumulative += count
                if cumulative >= position:
                    return float(rank)
            return float(histogram[-1][0])

        return (value_at(lower_position) + value_at(upper_position)) / 2.0

    def _rank_topk(self, k: int) -> Optional[float]:
        rank_count = self._mean_counts.get(_key("routing", "rank"), 0.0)
        if rank_count <= 0:
            return None
        prefix = _key("routing", "rank_hist") + "\x1f"
        topk = sum(
            count
            for key, count in self._totals.items()
            if key.startswith(prefix) and int(key.rsplit("\x1f", 1)[-1]) <= k
        )
        return topk / rank_count

    def format_routing_member_table(self) -> str:
        percent_metrics = (
            "selected_cond_is_gt",
            "selected_cond_is_bg",
            "selected_cond_is_other",
            "selected_group_contains_good7",
            "selected_group_same_as_oracle_group",
            "success7",
            "wrong_group7",
            "wrong_member7",
            "no_good_query7",
        )
        values = [
            _format_number(
                self._mean(_key("routing", "selected_cond_probability")),
                digits=4,
            ),
            _format_number(
                self._mean(_key("routing", "selected_cond_margin")),
                digits=4,
            ),
        ]
        values.extend(
            _format_number(
                self._mean(_key("routing", metric)),
                digits=2,
                percent=True,
            )
            for metric in percent_metrics
        )
        values.extend(
            (
                _format_number(self._mean(_key("routing", "rank")), digits=3),
                _format_number(self._rank_median(), digits=3),
                _format_number(self._rank_topk(1), digits=2, percent=True),
                _format_number(self._rank_topk(3), digits=2, percent=True),
                _format_number(self._rank_topk(5), digits=2, percent=True),
                _format_number(
                    self._mean(_key("routing", "within_group_member_gap_giou")),
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("routing", "cross_group_routing_gap_giou")),
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "variant_available7")),
                    digits=2,
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "variant_selected7")),
                    digits=2,
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "rescue_opportunity7")),
                    digits=2,
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "rescue7")),
                    digits=2,
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "matched_good_lost7")),
                    digits=2,
                    percent=True,
                ),
                _format_number(
                    self._mean(_key("idea", "good_concentration7")),
                    digits=4,
                ),
                _format_number(
                    self._mean(_key("idea", "best_group_purity7")),
                    digits=4,
                ),
                _format_number(
                    self._mean(_key("idea", "cond_iou_spearman")),
                    digits=4,
                ),
                _format_number(
                    self._mean_counts.get(
                        _key("idea", "cond_iou_spearman"),
                        0.0,
                    ),
                    integer=True,
                ),
            )
        )
        return _markdown_table(ROUTING_MEMBER_COLUMNS, [values])

    def _multigt_tasks(self) -> Sequence[str]:
        prefix = "multigt\x1f"
        tasks = {
            key.split("\x1f", 2)[1]
            for mapping in (self._totals, self._mean_sums)
            for key in mapping
            if key.startswith(prefix)
        }
        return sorted(tasks)

    def format_multigt_table(self) -> str:
        tasks = self._multigt_tasks()
        rows = []
        for task in tasks:
            prefix = _key("multigt", task)
            raw_names = (
                "targets",
                "conditions",
                "groups",
                "matched_groups",
                "bg_groups",
            )
            row = [task]
            row.extend(
                _format_number(self._totals.get(_key(prefix, name)), integer=True)
                for name in raw_names
            )
            row.append(
                _format_number(
                    self._ratio(
                        _key(prefix, "same_cond_multi_gt_samples", "numerator"),
                        _key(prefix, "same_cond_multi_gt_samples", "denominator"),
                    ),
                    digits=2,
                    percent=True,
                )
            )
            row.append(
                _format_number(
                    self._totals.get(_key(prefix, "unmatched_targets")),
                    integer=True,
                )
            )
            for name in (
                "group_recall50",
                "group_recall70",
                "duplicate_prediction_rate",
            ):
                row.append(
                    _format_number(
                        self._ratio(
                            _key(prefix, name, "numerator"),
                            _key(prefix, name, "denominator"),
                        ),
                        digits=4,
                    )
                )
            rows.append(row)
        if not rows:
            rows = [("N/A",) + ("N/A",) * (len(MULTIGT_COLUMNS) - 1)]
        return _markdown_table(MULTIGT_COLUMNS, rows)

    def format_multi_gt_table(self) -> str:
        """Alias with an explicit word boundary for multi-GT."""

        return self.format_multigt_table()

    def format_stage_table(self) -> str:
        metric_names = STAGE_COLUMNS[1:]
        rows = []
        for stage in range(self.num_stages):
            prefix = _key("stage", stage)
            rows.append(
                (f"st{stage}",)
                + tuple(
                    _format_number(self._mean(_key(prefix, metric)), digits=6)
                    for metric in metric_names
                )
            )
        return _markdown_table(STAGE_COLUMNS, rows)

    def format_stagewise_internal_table(self) -> str:
        """Format GT-backed stage trajectories; unavailable cells stay N/A."""

        rows = []
        for stage in range(self.num_stages):
            prefix = _key("stage_internal", stage)
            rows.append(
                (
                    f"st{stage}",
                    _format_number(
                        self._mean(_key(prefix, "query_oracle_giou")),
                        percent=True,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "no_good_query7")),
                        digits=2,
                        percent=True,
                    ),
                    _format_number(
                        self._mean(
                            _key(
                                prefix,
                                "selected_group_contains_good7",
                            )
                        ),
                        digits=2,
                        percent=True,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "routing_gap_giou")),
                        percent=True,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "cond_iou_spearman")),
                        digits=4,
                    ),
                    _format_number(
                        self._mean_counts.get(
                            _key(prefix, "cond_iou_spearman"),
                            0.0,
                        ),
                        integer=True,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "good_concentration7")),
                        digits=4,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "best_group_purity7")),
                        digits=4,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "wrong_member7")),
                        digits=2,
                        percent=True,
                    ),
                    _format_number(
                        self._mean(_key(prefix, "member_gap_giou")),
                        percent=True,
                    ),
                )
            )
        return _markdown_table(STAGEWISE_INTERNAL_COLUMNS, rows)

    @staticmethod
    def _json_count(value: float) -> Union[int, float]:
        rounded = round(value)
        return int(rounded) if abs(value - rounded) < 1e-9 else float(value)

    def _mean_record(self, key: str) -> Dict[str, Optional[float]]:
        count = self._mean_counts.get(key, 0.0)
        return {
            "value": self._mean(key),
            "valid_count": self._json_count(count),
        }

    def _ratio_record(
        self,
        numerator_key: str,
        denominator_key: str,
    ) -> Dict[str, Optional[float]]:
        numerator = self._totals.get(numerator_key, 0.0)
        denominator = self._totals.get(denominator_key, 0.0)
        return {
            "value": (
                numerator / denominator if denominator > 0 else None
            ),
            "numerator": self._json_count(numerator),
            "denominator": self._json_count(denominator),
        }

    def to_dict(self) -> Dict[str, object]:
        """Return a stable, JSON-serializable diagnostic summary.

        Values are fractions/IoU units in ``[0,1]`` unless a definition says
        otherwise.  Every conditional or validity-filtered mean carries its
        exact globally additive ``valid_count``.  This method is side-effect
        free and is intended to be called after :meth:`all_reduce_` when DDP is
        active.  It is a final report payload, not a serialization of the
        private accumulator: merge rank-local accumulators before calling this
        method rather than attempting to merge emitted JSON summaries.
        """

        baseline_ciou, baseline_giou = self._variant_metrics(
            "selected_query_mask"
        )
        variant_names = list(REFSEG_VARIANTS)
        variant_names.extend(
            sorted(set(self._variant_gt_select) - set(variant_names))
        )
        variants = OrderedDict()
        for variant in variant_names:
            prefix = _key("refseg", variant)
            intersection = self._totals.get(
                _key(prefix, "intersection"),
                0.0,
            )
            union = self._totals.get(_key(prefix, "union"), 0.0)
            ciou, giou = self._variant_metrics(variant)
            giou_count = self._mean_counts.get(_key(prefix, "giou"), 0.0)
            variants[variant] = {
                "cIoU": ciou,
                "gIoU": giou,
                "delta_cIoU_vs_selected_query": (
                    ciou - baseline_ciou
                    if ciou is not None and baseline_ciou is not None
                    else None
                ),
                "delta_gIoU_vs_selected_query": (
                    giou - baseline_giou
                    if giou is not None and baseline_giou is not None
                    else None
                ),
                "improved_rate_vs_selected_query": self._mean(
                    _key(prefix, "improved")
                ),
                "worsened_rate_vs_selected_query": self._mean(
                    _key(prefix, "worsened")
                ),
                "gt_selected": bool(
                    self._variant_gt_select.get(variant, False)
                ),
                "sufficient_statistics": {
                    "intersection": float(intersection),
                    "union": float(union),
                    "sample_iou_sum": float(
                        self._mean_sums.get(_key(prefix, "giou"), 0.0)
                    ),
                    "sample_count": self._json_count(giou_count),
                },
            }

        structures = []
        for stage in range(self.num_stages):
            prefix = _key("structure", stage)
            samples = self._totals.get(_key(prefix, "samples"), 0.0)
            groups = self._totals.get(_key(prefix, "groups"), 0.0)
            members = self._totals.get(_key(prefix, "members"), 0.0)
            structures.append(
                {
                    "stage": stage,
                    "samples": self._json_count(samples),
                    "avg_groups": (
                        groups / samples if samples > 0 else None
                    ),
                    "avg_group_size": (
                        members / groups if groups > 0 else None
                    ),
                    "singleton_rate": self._ratio(
                        _key(prefix, "singletons"),
                        _key(prefix, "groups"),
                    ),
                    "largest_group": self._maxima.get(
                        _key(prefix, "largest")
                    ),
                    "group_purity": self._mean_record(
                        _key(prefix, "purity")
                    ),
                    "good_group_count7": self._mean_record(
                        _key(prefix, "good_group_count7")
                    ),
                    "duplicate_group_count7": self._mean_record(
                        _key(prefix, "duplicate_group_count7")
                    ),
                    "bridge_rate": self._mean_record(
                        _key(prefix, "bridge")
                    ),
                }
            )

        routing_mean_names = (
            "selected_cond_probability",
            "selected_cond_margin",
            "selected_cond_is_gt",
            "selected_cond_is_bg",
            "selected_cond_is_other",
            "selected_group_contains_good7",
            "selected_group_same_as_oracle_group",
            "success7",
            "wrong_group7",
            "wrong_member7",
            "no_good_query7",
            "within_group_member_gap_giou",
            "cross_group_routing_gap_giou",
        )
        routing = {
            name: self._mean_record(_key("routing", name))
            for name in routing_mean_names
        }
        routing["first_good_group_rank"] = {
            **self._mean_record(_key("routing", "rank")),
            "median": self._rank_median(),
            "top1_rate": self._rank_topk(1),
            "top3_rate": self._rank_topk(3),
            "top5_rate": self._rank_topk(5),
        }
        routing["definitions"] = {
            "selected_group": (
                "valid Group with the highest softmax probability for the "
                "effective sample's formal target Cond; ties use the lowest "
                "physical root"
            ),
            "selected_cond_probability": (
                "selected Group's softmax probability for the formal target "
                "Cond after invalid condition columns are masked"
            ),
            "selected_cond_margin": (
                "selected Group target-Cond logit minus its explicit "
                "background logit; this is a logit margin, not a probability "
                "difference"
            ),
            "selected_cond_is_gt": (
                "selected Group's valid-class argmax is the formal target Cond"
            ),
            "selected_cond_is_bg": (
                "selected Group's valid-class argmax is explicit background"
            ),
            "selected_cond_is_other": (
                "selected Group's valid-class argmax is another Cond"
            ),
            "selected_group_contains_good7": (
                "selected Group contains at least one Query with IoU>=0.7"
            ),
            "selected_group_same_as_oracle_group": (
                "selected Group contains the global max-IoU Query"
            ),
            "success7": (
                "formal selected-Group quality representative has IoU>=0.7"
            ),
            "wrong_group7": (
                "a good Query exists globally but the selected Group contains "
                "none"
            ),
            "wrong_member7": (
                "the selected Group contains a good Query but its formal "
                "quality representative is below IoU 0.7"
            ),
            "no_good_query7": "no Query among all 200 reaches IoU 0.7",
            "first_good_group_rank": (
                "one-based target-Cond score rank of the first Group "
                "containing a good Query; undefined for no-good samples"
            ),
            "within_group_member_gap_giou": (
                "selected-Group oracle-member IoU minus formal quality "
                "representative IoU, averaged per sample"
            ),
            "cross_group_routing_gap_giou": (
                "global 200-Query oracle IoU minus selected-Group "
                "oracle-member IoU, averaged per sample"
            ),
        }

        idea_names = (
            "variant_available7",
            "variant_selected7",
            "rescue_opportunity7",
            "rescue7",
            "matched_good_lost7",
            "good_concentration7",
            "best_group_purity7",
            "cond_iou_spearman",
        )
        idea = {
            name: self._mean_record(_key("idea", name))
            for name in idea_names
        }
        idea["definitions"] = {
            "variant_available7": (
                "sample has an IoU>=0.7 Query other than the final-stage "
                "deterministic full-resolution evaluation matcher's selected "
                "member; this matcher reuses the configured training costs "
                "but is not a historical stochastic training assignment"
            ),
            "variant_selected7": (
                "formal quality representative is IoU>=0.7 and differs from "
                "the deterministic full-resolution evaluation matcher's "
                "selected member"
            ),
            "rescue_opportunity7": (
                "evaluation-matched member is below 0.7 while a variant "
                "Query is good"
            ),
            "rescue7": (
                "conditional on rescueOpportunity7: formal representative "
                "reaches IoU>=0.7"
            ),
            "matched_good_lost7": (
                "conditional on a good evaluation-matched member: formal "
                "representative falls below IoU 0.7"
            ),
            "good_concentration7": (
                "largest number of good Queries in one Group divided by all "
                "good Queries in that sample"
            ),
            "best_group_purity7": (
                "good-Query fraction inside the Group attaining the largest "
                "good-Query count (lowest root breaks ties)"
            ),
            "cond_iou_spearman": (
                "per-sample Spearman correlation between formal target-Cond "
                "Group probability and Group oracle member IoU"
            ),
        }

        multigt = OrderedDict()
        for task in self._multigt_tasks():
            prefix = _key("multigt", task)
            task_summary = {
                name: self._json_count(
                    self._totals.get(_key(prefix, name), 0.0)
                )
                for name in (
                    "targets",
                    "conditions",
                    "groups",
                    "matched_groups",
                    "bg_groups",
                    "unmatched_targets",
                )
            }
            for name in (
                "same_cond_multi_gt_samples",
                "group_recall50",
                "group_recall70",
                "duplicate_prediction_rate",
            ):
                task_summary[name] = self._ratio_record(
                    _key(prefix, name, "numerator"),
                    _key(prefix, name, "denominator"),
                )
            multigt[task] = task_summary

        stages = []
        for stage in range(self.num_stages):
            prefix = _key("stage", stage)
            stages.append(
                {
                    "stage": stage,
                    **{
                        metric: self._mean_record(_key(prefix, metric))
                        for metric in STAGE_COLUMNS[1:]
                    },
                }
            )

        stagewise_internal = []
        for stage in range(self.num_stages):
            prefix = _key("stage_internal", stage)
            stagewise_internal.append(
                {
                    "stage": stage,
                    **{
                        metric: self._mean_record(_key(prefix, metric))
                        for metric in (
                            "query_oracle_giou",
                            "no_good_query7",
                            "selected_group_contains_good7",
                            "routing_gap_giou",
                            "cond_iou_spearman",
                            "good_concentration7",
                            "best_group_purity7",
                            "wrong_member7",
                            "member_gap_giou",
                        )
                    },
                }
            )

        final_structure_prefix = _key(
            "structure",
            self.num_stages - 1,
        )
        diagnostic_samples = self._totals.get(
            _key(final_structure_prefix, "samples"),
            0.0,
        )
        diagnostic_members = self._totals.get(
            _key(final_structure_prefix, "members"),
            0.0,
        )
        queries = (
            diagnostic_members / diagnostic_samples
            if diagnostic_samples > 0
            else None
        )
        partition_names = (
            "success7",
            "no_good_query7",
            "wrong_group7",
            "wrong_member7",
        )
        partition_values = [
            self._mean(_key("routing", name)) for name in partition_names
        ]
        partition_counts = [
            self._mean_counts.get(_key("routing", name), 0.0)
            for name in partition_names
        ]
        partition_sum = (
            sum(value for value in partition_values if value is not None)
            if all(value is not None for value in partition_values)
            else None
        )
        partition_valid_count = (
            partition_counts[0]
            if partition_counts
            and all(
                abs(count - partition_counts[0]) < 1e-9
                for count in partition_counts
            )
            else None
        )

        return {
            "schema_version": 1,
            "metric_unit": "fraction",
            "summary_contract": (
                "final reduced report with reported validity denominators; "
                "not a mergeable rank-local accumulator serialization"
            ),
            "num_stages": self.num_stages,
            "globally_reduced": bool(self._globally_reduced),
            "formal_variant_name": "selected_group_quality_query_mask",
            "sanity": {
                "stages": self.num_stages,
                "queries": (
                    self._json_count(queries)
                    if queries is not None
                    else None
                ),
                "global_diagnostic_samples": self._json_count(
                    diagnostic_samples
                ),
                "routing_partition_sum7": partition_sum,
                "routing_partition_valid_count": (
                    self._json_count(partition_valid_count)
                    if partition_valid_count is not None
                    else None
                ),
            },
            "refseg_variants": variants,
            "group_structure": structures,
            "routing_member": routing,
            "idea_diagnostics": idea,
            "multi_gt": multigt,
            "stage": stages,
            "stagewise_internal": {
                "definitions": {
                    "query_oracle_giou": (
                        "mean per-sample max IoU over all 200 Queries"
                    ),
                    "no_good_query7": (
                        "sample has no Query with IoU>=0.7"
                    ),
                    "selected_group_contains_good7": (
                        "target-Cond top-scoring Group contains an "
                        "IoU>=0.7 Query"
                    ),
                    "routing_gap_giou": (
                        "global 200-Query oracle IoU minus selected-Group "
                        "oracle-member IoU, averaged per sample"
                    ),
                    "cond_iou_spearman": (
                        "per-sample Spearman correlation between target-Cond "
                        "Group probability and Group oracle-member IoU"
                    ),
                    "good_concentration7": (
                        "largest good-Query count in one Group divided by "
                        "all good Queries"
                    ),
                    "best_group_purity7": (
                        "good-Query fraction in the Group attaining the "
                        "largest good-Query count"
                    ),
                    "wrong_member7": (
                        "selected Group contains a good Query but its formal "
                        "quality representative is below IoU 0.7; final "
                        "stage only"
                    ),
                    "member_gap_giou": (
                        "selected-Group oracle-member IoU minus formal "
                        "quality-representative IoU; final stage only"
                    ),
                },
                "stages": stagewise_internal,
            },
        }

    def summary(self) -> Dict[str, object]:
        """Alias for :meth:`to_dict` for evaluator-facing callers."""

        return self.to_dict()

    def terminal_tables(self) -> "OrderedDict[str, str]":
        """Return the five formal terminal tables in a stable order."""

        return OrderedDict(
            (
                ("refseg", self.format_refseg_table()),
                ("group_structure", self.format_group_structure_table()),
                ("routing_member", self.format_routing_member_table()),
                ("multigt", self.format_multigt_table()),
                (
                    "stagewise_internal",
                    self.format_stagewise_internal_table(),
                ),
            )
        )

    def format_terminal_tables(self, include_stage: bool = True) -> str:
        """Format tables for rank-zero logging; this method never prints."""

        titled = [
            f"[{name}]\n{table}" for name, table in self.terminal_tables().items()
        ]
        if include_stage:
            titled.append(f"[stage]\n{self.format_stage_table()}")
        return "\n\n".join(titled)


__all__ = [
    "GROUP_STRUCTURE_COLUMNS",
    "GT_SELECT_VARIANTS",
    "GroupDiagnostics",
    "MULTIGT_COLUMNS",
    "REFSEG_COLUMNS",
    "REFSEG_VARIANTS",
    "ROUTING_MEMBER_COLUMNS",
    "STAGE_COLUMNS",
    "STAGEWISE_INTERNAL_COLUMNS",
]
