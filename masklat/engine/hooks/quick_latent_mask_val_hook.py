"""Opt-in four-val mask evaluation for the short latent S3 adaptation run."""

from copy import deepcopy
import math
import os.path as osp

from .periodic_refseg_eval_hook import (
    BUILDER,
    PeriodicRefSegEvalHook,
    _raise_on_rank_errors,
    evaluate_dataset,
    get_rank,
    get_world_size,
)


VAL_NAMES = (
    "refcoco_val_refseg", "refcoco+_val_refseg", "refcocog_val_refseg", "val_reaseg",
)


def _validate_summary(summary, data_name, world_size):
    """Only corrected all-rank foreground metrics are the comparison result."""
    if not isinstance(summary, dict) or summary.get("data_name") != data_name:
        raise ValueError(f"missing/mismatched validation summary for {data_name}")
    count = summary.get("num_predictions")
    if type(count) is not int or count <= 0:
        raise ValueError(f"invalid prediction count for {data_name}: {count!r}")
    foreground = "reason" if data_name == "val_reaseg" else "refer"
    metrics = summary.get("metrics", {}).get(foreground, {})
    audit = summary.get("aggregation_audit", {}).get("correct_all_ranks", {})
    if (summary.get("schema_version") != 3
            or audit.get("world_size") != world_size
            or audit.get("num_predictions") != count
            or audit.get("foreground_class") != foreground):
        raise ValueError(f"missing/mismatched all-rank audit for {data_name}")
    for name in ("cIoU", "gIoU"):
        value = metrics.get(name)
        audited = audit.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 100
                or isinstance(audited, bool) or not isinstance(audited, (int, float))
                or not math.isfinite(audited)
                or not math.isclose(value, audited, rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f"invalid all-rank {foreground}/{name} for {data_name}")
    return foreground, metrics


class QuickLatentMaskValHook(PeriodicRefSegEvalHook):
    """No periodic work; optional initial suite plus one completed-run suite.

    The historical parent hook only summarizes the ``refer`` class. This
    isolated hook also supports ReaSeg's ``reason`` class without changing the
    original hook. Dataset objects are reused for initial/final evaluation to
    avoid re-registering conflicting temporary GT paths in MetadataCatalog.
    """

    def __init__(self, datasets, evaluators, eval_before_train=False,
                 dataloader_num_workers=1):
        if not isinstance(datasets, (list, tuple)) or len(datasets) != 4:
            raise ValueError("quick mask validation requires exactly four datasets")
        if not isinstance(evaluators, (list, tuple)) or len(evaluators) != 4:
            raise ValueError("quick mask validation requires exactly four evaluators")
        names = tuple(self._config_data_name(d, e) for d, e in zip(datasets, evaluators))
        if names != VAL_NAMES:
            raise ValueError(f"quick validation must use {VAL_NAMES}, got {names}")
        super().__init__(
            dataset=datasets[0], evaluator=evaluators[0], interval=1,
            dataloader_num_workers=dataloader_num_workers,
            eval_at_start=eval_before_train, final_datasets=datasets,
            final_evaluators=evaluators, final_output_subdir="final_val4")
        self._suite_datasets = {}

    def before_train(self, runner):
        if self.eval_at_start and int(runner.iter) == 0:
            self._run_suite(runner, step=0, phase="initial_val4", checkpoint=None)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # Explicitly disable periodic evaluation, regardless of interval/steps.
        return

    def _evaluate_final(self, runner, *, step, checkpoint):
        # Inherit the parent's completed-run guard in after_train. The label
        # names the current in-memory state; export timing belongs to the
        # original save hook, so do not claim a checkpoint file exists here.
        self._run_suite(runner, step=step, phase="final_val4", checkpoint=None)

    def _run_suite(self, runner, *, step, phase, checkpoint):
        runtime_model = runner.model
        model = self._unwrap_model(runtime_model)
        llm_config = getattr(getattr(model, "llm", None), "config", None)
        old_cache = getattr(llm_config, "use_cache", None)
        old_postprocess = model.postprocess_fn
        was_training = runtime_model.training
        rank, world_size = get_rank(), get_world_size()
        output_root = osp.join(runner.work_dir, phase)
        summaries = []
        runner.logger.info(f"Quick latent mask validation: {phase}, four val splits")
        try:
            if old_cache is not None:
                llm_config.use_cache = False
            runtime_model.eval()
            for dataset_cfg, evaluator_cfg in zip(
                    self.final_dataset_cfgs, self.final_evaluator_cfgs):
                name = self._config_data_name(dataset_cfg, evaluator_cfg)
                dataset = evaluator = summary = None
                error = None
                try:
                    if name not in self._suite_datasets:
                        self._suite_datasets[name] = BUILDER.build(deepcopy(dataset_cfg))
                    dataset = self._suite_datasets[name]
                    evaluator = BUILDER.build(deepcopy(evaluator_cfg))
                    evaluator.metadata = dataset.metadata
                    evaluator.output_dir = osp.join(output_root, "pred_data", name)
                    model.postprocess_fn = dataset.postprocess_fn
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                _raise_on_rank_errors(error, phase=f"{phase} construction", data_name=name)
                error = None
                try:
                    summary = evaluate_dataset(
                        model=model, dataset=dataset, evaluator=evaluator,
                        rank=rank, world_size=world_size, data_name=name,
                        dataloader_num_workers=self.dataloader_num_workers,
                        collect_topology_diagnostics=False)
                    if rank == 0:
                        foreground, metrics = _validate_summary(summary, name, world_size)
                        enriched = dict(summary, step=int(step), world_size=world_size,
                                        checkpoint=checkpoint, model_state="in_memory",
                                        phase=phase)
                        self._atomic_write_json(
                            osp.join(evaluator.output_dir, "summary.json"), enriched)
                        summaries.append(enriched)
                        runner.logger.info(
                            f"[{name}] all-rank {foreground}: "
                            f"cIoU={metrics['cIoU']:.2f}, gIoU={metrics['gIoU']:.2f}")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                _raise_on_rank_errors(error, phase=f"{phase} evaluation", data_name=name)
                dataset = evaluator = summary = None
                self._release_cuda_cache()
            error = None
            try:
                if rank == 0:
                    if tuple(s["data_name"] for s in summaries) != VAL_NAMES:
                        raise RuntimeError("incomplete four-val summaries")
                    self._atomic_write_json(osp.join(output_root, "summary.json"), dict(
                        schema_version=1, report_type="quick_latent_mask_val4",
                        phase=phase, step=int(step), world_size=world_size,
                        model_state="in_memory", checkpoint=checkpoint,
                        initialization_checkpoint=getattr(
                            model, "_cascade_s3_checkpoint",
                            getattr(model, "_no_latent_update_s3_checkpoint", None)),
                        dataset_order=list(VAL_NAMES), num_datasets=4,
                        datasets=summaries))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _raise_on_rank_errors(error, phase=f"{phase} summary", data_name="val4")
        finally:
            model.postprocess_fn = old_postprocess
            if old_cache is not None:
                llm_config.use_cache = old_cache
            runtime_model.train(was_training)
            self._cleanup_before_next_train_iter = False
            self._release_cuda_cache()


__all__ = ["QuickLatentMaskValHook"]
