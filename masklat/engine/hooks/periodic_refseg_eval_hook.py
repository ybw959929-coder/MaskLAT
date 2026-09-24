import gc
import json
import math
import os
import os.path as osp
from copy import deepcopy

import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from xtuner.registry import BUILDER

from masklat.tools.eval import (
    _raise_on_rank_errors,
    evaluate_dataset,
)


class PeriodicRefSegEvalHook(Hook):
    """Run one existing distributed RefSeg evaluator during training.

    This hook deliberately reuses ``masklat.tools.eval.evaluate_dataset`` instead
    of MMEngine's generic ValLoop: MaskLAT's segmentation evaluators consume
    postprocessed masks and dataset metadata through a custom interface.
    """

    # CheckpointHook uses VERY_LOW (90).  Run immediately afterwards so step
    # 500 is recoverable even if validation itself fails.
    priority = 95

    def __init__(
        self,
        dataset,
        evaluator,
        interval=500,
        dataloader_num_workers=1,
        eval_at_start=False,
        final_datasets=None,
        final_evaluators=None,
        final_output_subdir="final_refseg",
    ):
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError(f"interval must be a positive integer, got {interval}")
        if not isinstance(dataloader_num_workers, int) or dataloader_num_workers < 0:
            raise ValueError(
                "dataloader_num_workers must be a non-negative integer, "
                f"got {dataloader_num_workers}"
            )
        if not isinstance(eval_at_start, bool):
            raise TypeError(
                f"eval_at_start must be a bool, got {type(eval_at_start).__name__}"
            )
        if (final_datasets is None) != (final_evaluators is None):
            raise ValueError(
                "final_datasets and final_evaluators must be provided together"
            )
        if final_datasets is not None:
            if not isinstance(final_datasets, (list, tuple)) or not (
                isinstance(final_evaluators, (list, tuple))
            ):
                raise TypeError(
                    "final_datasets/final_evaluators must be lists or tuples"
                )
            if not final_datasets or len(final_datasets) != len(final_evaluators):
                raise ValueError(
                    "final_datasets/final_evaluators must be non-empty and "
                    "have equal lengths"
                )
        if not isinstance(final_output_subdir, str) or not final_output_subdir:
            raise ValueError("final_output_subdir must be a non-empty string")
        normalized_final_subdir = osp.normpath(final_output_subdir)
        if osp.isabs(normalized_final_subdir) or normalized_final_subdir.startswith(
            ".."
        ):
            raise ValueError(
                "final_output_subdir must stay inside runner.work_dir"
            )
        self.dataset_cfg = deepcopy(dataset)
        self.evaluator_cfg = deepcopy(evaluator)
        self.interval = interval
        self.dataloader_num_workers = dataloader_num_workers
        self.eval_at_start = eval_at_start
        self._dataset = None
        self.final_dataset_cfgs = (
            None if final_datasets is None else deepcopy(list(final_datasets))
        )
        self.final_evaluator_cfgs = (
            None if final_evaluators is None else deepcopy(list(final_evaluators))
        )
        if self.final_dataset_cfgs is not None:
            final_names = [
                self._config_data_name(dataset_cfg, evaluator_cfg)
                for dataset_cfg, evaluator_cfg in zip(
                    self.final_dataset_cfgs,
                    self.final_evaluator_cfgs,
                )
            ]
            if len(set(final_names)) != len(final_names):
                raise ValueError(
                    f"final RefSeg data_name values must be unique: {final_names}"
                )
        self.final_output_subdir = normalized_final_subdir
        # ``after_train_iter`` still owns the preceding training batch and
        # outputs while periodic evaluation is running.  Its final cleanup is
        # therefore necessarily too early to release every allocator block.
        # Repeat the cleanup at the next ``before_train_iter``, after that hook
        # stack and the previous batch have gone out of scope.
        self._cleanup_before_next_train_iter = False

    @staticmethod
    def _atomic_write_json(output_path, payload):
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        temporary_path = output_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
        os.replace(temporary_path, output_path)

    @staticmethod
    def _config_data_name(dataset_cfg, evaluator_cfg):
        evaluator_name = (
            evaluator_cfg.get("data_name")
            if isinstance(evaluator_cfg, dict)
            else getattr(evaluator_cfg, "data_name", None)
        )
        dataset_name = (
            dataset_cfg.get("data_name")
            if isinstance(dataset_cfg, dict)
            else getattr(dataset_cfg, "data_name", None)
        )
        data_name = evaluator_name or dataset_name
        if not isinstance(data_name, str) or not data_name:
            raise ValueError("every final evaluator requires a data_name")
        if (
            evaluator_name is not None
            and dataset_name is not None
            and evaluator_name != dataset_name
        ):
            raise ValueError(
                "final dataset/evaluator data_name mismatch: "
                f"{dataset_name!r} != {evaluator_name!r}"
            )
        return data_name

    @staticmethod
    def _release_cuda_cache():
        gc.collect()
        if torch.cuda.is_available():
            # Evaluation and distributed collectives enqueue asynchronous work.
            # Make blocks reclaimable before returning them to the CUDA driver.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @staticmethod
    def _unwrap_model(model):
        while is_model_wrapper(model):
            model = model.module
        return model

    def _build_dataset(self):
        if self._dataset is not None:
            return self._dataset

        local_error = None
        try:
            self._dataset = BUILDER.build(self.dataset_cfg)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _raise_on_rank_errors(
            local_error,
            phase="periodic dataset construction",
            data_name="refcoco_val_refseg",
        )
        return self._dataset

    def _evaluate(self, runner, *, step, checkpoint):
        # Change the state of the runtime wrapper (for example DeepSpeed), not
        # only its innermost module.  The unwrapped MaskLAT module is still what
        # the custom evaluator consumes.
        runtime_model = runner.model
        model = self._unwrap_model(runtime_model)
        dataset = self._build_dataset()
        evaluator = None
        local_error = None

        try:
            evaluator = BUILDER.build(self.evaluator_cfg)
            evaluator.metadata = dataset.metadata
            evaluator.output_dir = osp.join(
                runner.work_dir,
                "periodic_val",
                f"iter_{step:09d}",
                evaluator.data_name,
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _raise_on_rank_errors(
            local_error,
            phase="periodic evaluator construction",
            data_name="refcoco_val_refseg",
        )

        was_training = runtime_model.training
        llm = getattr(model, "llm", None)
        old_use_cache = (
            getattr(getattr(llm, "config", None), "use_cache", None)
            if llm is not None
            else None
        )
        old_postprocess_fn = model.postprocess_fn

        runner.logger.info(
            "Running distributed refcoco_val_refseg evaluation at "
            f"train iteration {step}."
        )
        try:
            if llm is not None and old_use_cache is not None:
                # Periodic RefCOCO validation is teacher-forced (``mode=tensor``),
                # not autoregressive generation.  A KV cache is unused here and
                # creates large evaluation-only allocation shapes that fragment
                # the subsequent training backward pass.
                llm.config.use_cache = False
            # Do not disable/re-enable activation checkpointing here.  Every
            # MaskLAT checkpoint branch already requires ``self.training``;
            # eval mode therefore bypasses it naturally.  Transformers'
            # ``gradient_checkpointing_enable`` may register input-gradient
            # hooks, so toggling it every 500 steps can mutate/accumulate
            # training state across periodic evaluations.
            runtime_model.eval()
            model.postprocess_fn = dataset.postprocess_fn
            summary = evaluate_dataset(
                model=model,
                dataset=dataset,
                evaluator=evaluator,
                rank=get_rank(),
                world_size=get_world_size(),
                data_name=evaluator.data_name,
                dataloader_num_workers=self.dataloader_num_workers,
                collect_topology_diagnostics=False,
            )
            if get_rank() == 0:
                if not isinstance(summary, dict) or not summary:
                    raise RuntimeError(
                        "periodic refcoco_val_refseg evaluation did not "
                        "return a rank-0 summary"
                    )
                summary = {
                    **summary,
                    "step": step,
                    "world_size": get_world_size(),
                    "checkpoint": checkpoint,
                }
                summary_path = osp.join(evaluator.output_dir, "summary.json")
                latest_summary_path = osp.join(
                    runner.work_dir,
                    "periodic_val",
                    "latest_summary.json",
                )
                for output_path in (summary_path, latest_summary_path):
                    self._atomic_write_json(output_path, summary)

                refer_metrics = summary["metrics"]["refer"]
                runner.logger.info(
                    "[REFCOCO_VAL] "
                    f"step={step} "
                    f"predictions={summary['num_predictions']} "
                    f"refer_cIoU={refer_metrics['cIoU']:.2f} "
                    f"refer_gIoU={refer_metrics['gIoU']:.2f} "
                    f"summary={summary_path}"
                )
        finally:
            model.postprocess_fn = old_postprocess_fn
            if llm is not None and old_use_cache is not None:
                llm.config.use_cache = old_use_cache
            runtime_model.train(was_training)
            # Evaluation uses different tensor shapes from mixed-data training.
            # Drop the per-evaluation accumulator and return unused allocator
            # blocks before the next high-watermark training batch.  This is
            # fragmentation hygiene; bounded quality-IoU chunking remains the
            # structural protection against giant topology groups.
            evaluator = None
            self._cleanup_before_next_train_iter = True
            self._release_cuda_cache()

    def _evaluate_final(self, runner, *, step, checkpoint):
        """Evaluate all configured RefSeg splits and write one total report."""

        if self.final_dataset_cfgs is None:
            return
        runtime_model = runner.model
        model = self._unwrap_model(runtime_model)
        llm = getattr(model, "llm", None)
        old_use_cache = (
            getattr(getattr(llm, "config", None), "use_cache", None)
            if llm is not None
            else None
        )
        old_postprocess_fn = model.postprocess_fn
        was_training = runtime_model.training
        output_root = osp.join(runner.work_dir, self.final_output_subdir)
        rank = get_rank()
        world_size = get_world_size()
        summaries = []

        cached_periodic_data_name = None
        if self._dataset is not None:
            cached_periodic_data_name = getattr(
                self._dataset,
                "data_name",
                None,
            )
            if not cached_periodic_data_name:
                cached_periodic_data_name = self._config_data_name(
                    self.dataset_cfg,
                    self.evaluator_cfg,
                )

        runner.logger.info(
            "Training is complete; running distributed evaluation on "
            f"{len(self.final_dataset_cfgs)} RefSeg datasets."
        )
        try:
            if llm is not None and old_use_cache is not None:
                llm.config.use_cache = False
            runtime_model.eval()

            for dataset_cfg, evaluator_cfg in zip(
                self.final_dataset_cfgs,
                self.final_evaluator_cfgs,
            ):
                data_name = self._config_data_name(
                    dataset_cfg,
                    evaluator_cfg,
                )
                dataset = None
                evaluator = None
                summary = None
                local_error = None
                try:
                    # Periodic validation already owns the RefCOCO-val dataset
                    # and its registered temporary GT JSON.  Rebuilding the
                    # same data_name creates a different temporary path, which
                    # MetadataCatalog correctly rejects as conflicting
                    # metadata.  Reuse the cached dataset for that one split.
                    if data_name == cached_periodic_data_name:
                        dataset = self._dataset
                    else:
                        dataset = BUILDER.build(dataset_cfg)
                    evaluator = BUILDER.build(evaluator_cfg)
                    evaluator.metadata = dataset.metadata
                    evaluator.output_dir = osp.join(
                        output_root,
                        "pred_data",
                        data_name,
                    )
                    model.postprocess_fn = dataset.postprocess_fn
                except Exception as error:
                    local_error = f"{type(error).__name__}: {error}"
                _raise_on_rank_errors(
                    local_error,
                    phase="final RefSeg dataset/evaluator construction",
                    data_name=data_name,
                )

                runner.logger.info(f"Final evaluation: {data_name}")
                local_error = None
                try:
                    summary = evaluate_dataset(
                        model=model,
                        dataset=dataset,
                        evaluator=evaluator,
                        rank=rank,
                        world_size=world_size,
                        data_name=data_name,
                        dataloader_num_workers=(
                            self.dataloader_num_workers
                        ),
                        collect_topology_diagnostics=False,
                    )
                    if rank == 0:
                        if not isinstance(summary, dict) or not summary:
                            raise RuntimeError(
                                f"{data_name} returned no rank-0 summary"
                            )
                        if summary.get("data_name") != data_name:
                            raise ValueError(
                                f"{data_name} returned mismatched data_name "
                                f"{summary.get('data_name')!r}"
                            )
                        prediction_count = summary.get("num_predictions")
                        if (
                            not isinstance(prediction_count, int)
                            or isinstance(prediction_count, bool)
                            or prediction_count <= 0
                        ):
                            raise ValueError(
                                f"{data_name} has invalid num_predictions="
                                f"{prediction_count!r}"
                            )
                        refer = summary.get("metrics", {}).get("refer", {})
                        for metric_name in ("cIoU", "gIoU"):
                            metric = refer.get(metric_name)
                            if (
                                not isinstance(metric, (int, float))
                                or isinstance(metric, bool)
                                or not math.isfinite(float(metric))
                            ):
                                raise ValueError(
                                    f"{data_name} has invalid "
                                    f"refer/{metric_name}={metric!r}"
                                )
                        enriched_summary = {
                            **summary,
                            "step": int(step),
                            "world_size": int(world_size),
                            "checkpoint": checkpoint,
                        }
                        self._atomic_write_json(
                            osp.join(evaluator.output_dir, "summary.json"),
                            enriched_summary,
                        )
                        summaries.append(enriched_summary)
                except Exception as error:
                    local_error = f"{type(error).__name__}: {error}"
                _raise_on_rank_errors(
                    local_error,
                    phase="final RefSeg evaluation",
                    data_name=data_name,
                )

                dataset = None
                evaluator = None
                summary = None
                self._release_cuda_cache()

            local_error = None
            try:
                if rank == 0:
                    if len(summaries) != len(self.final_dataset_cfgs):
                        raise RuntimeError(
                            "rank-0 final summaries are incomplete: "
                            f"{len(summaries)} != "
                            f"{len(self.final_dataset_cfgs)}"
                        )
                    total_summary = {
                        "schema_version": 1,
                        "report_type": "final_refseg_evaluation",
                        "step": int(step),
                        "world_size": int(world_size),
                        "checkpoint": checkpoint,
                        "num_datasets": len(summaries),
                        "dataset_order": [
                            summary["data_name"] for summary in summaries
                        ],
                        "datasets": summaries,
                    }
                    total_path = osp.join(output_root, "summary.json")
                    self._atomic_write_json(total_path, total_summary)

                    table_rows = [
                        "| dataset | N | cIoU | gIoU |",
                        "| --- | ---: | ---: | ---: |",
                    ]
                    for summary in summaries:
                        refer = summary["metrics"]["refer"]
                        table_rows.append(
                            "| {name} | {count} | {ciou:.2f} | {giou:.2f} |".format(
                                name=summary["data_name"],
                                count=summary["num_predictions"],
                                ciou=refer["cIoU"],
                                giou=refer["gIoU"],
                            )
                        )
                    runner.logger.info(
                        "Final RefSeg Summary "
                        f"({len(summaries)} datasets)\n"
                        + "\n".join(table_rows)
                        + f"\nTotal summary: {total_path}"
                    )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            _raise_on_rank_errors(
                local_error,
                phase="final RefSeg summary writing",
                data_name="all_refseg",
            )
        finally:
            model.postprocess_fn = old_postprocess_fn
            if llm is not None and old_use_cache is not None:
                llm.config.use_cache = old_use_cache
            runtime_model.train(was_training)
            self._cleanup_before_next_train_iter = False
            self._release_cuda_cache()

    def before_train(self, runner):
        """Evaluate initialized weights once, but never repeat on resume."""
        if self.eval_at_start and runner.iter == 0:
            self._evaluate(runner, step=0, checkpoint=None)

    def before_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
    ):
        """Release evaluation-shaped blocks after the prior hook stack exits."""

        if not self._cleanup_before_next_train_iter:
            return
        self._release_cuda_cache()
        self._cleanup_before_next_train_iter = False

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ):
        if self.every_n_train_iters(runner, self.interval):
            step = runner.iter + 1
            max_iters = getattr(runner, "max_iters", None)
            if (
                self.final_dataset_cfgs is not None
                and max_iters is not None
                and int(step) >= int(max_iters)
            ):
                # ``after_train`` immediately evaluates the complete final
                # suite, which already contains the periodic RefCOCO-val
                # split.  Avoid evaluating that split twice at the same
                # terminal checkpoint when max_iters is interval-aligned.
                return
            self._evaluate(
                runner,
                step=step,
                checkpoint=f"iter_{step}.pth",
            )

    def after_train(self, runner):
        if self.final_dataset_cfgs is None:
            return
        max_iters = getattr(runner, "max_iters", None)
        if max_iters is not None and int(runner.iter) < int(max_iters):
            runner.logger.warning(
                "Skipping final RefSeg evaluation because training stopped "
                f"early at iter {int(runner.iter)}/{int(max_iters)}."
            )
            return
        step = int(runner.iter)
        self._evaluate_final(
            runner,
            step=step,
            checkpoint=f"iter_{step}.pth",
        )
