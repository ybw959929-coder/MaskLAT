import datetime
from numbers import Real

from mmengine.hooks import LoggerHook


class ConciseLoggerHook(LoggerHook):
    """Keep full scalar records while printing only critical train metrics."""

    terminal_metrics = (
        "loss",
        "loss_llm",
        "loss_seg",
        "grad_norm",
    )

    @staticmethod
    def _scalar(value):
        if isinstance(value, Real):
            return float(value)
        if hasattr(value, "numel") and value.numel() == 1:
            return float(value.item())
        return None

    @classmethod
    def _find_tag(cls, tags, name):
        if name in tags:
            return tags[name]
        hierarchical_name = f"train/{name}"
        return tags.get(hierarchical_name)

    @classmethod
    def _format_train_log(cls, runner, tags):
        step = runner.iter + 1
        parts = [f"Iter(train) [{step}/{runner.max_iters}]"]

        for name, value in tags.items():
            if name.rsplit("/", 1)[-1].endswith("lr"):
                scalar = cls._scalar(value)
                if scalar is not None:
                    parts.append(
                        f"{name.rsplit('/', 1)[-1]}: {scalar:.4e}"
                    )

        runtime_info = runner.message_hub.runtime_info
        if "eta" in runtime_info:
            eta = datetime.timedelta(seconds=int(runtime_info["eta"]))
            parts.append(f"eta: {eta}")

        for name in ("time", "data_time"):
            scalar = cls._scalar(cls._find_tag(tags, name))
            if scalar is not None:
                parts.append(f"{name}: {scalar:.4f}")

        memory = cls._scalar(cls._find_tag(tags, "memory"))
        if memory is not None:
            parts.append(f"memory: {int(memory)}")

        for name in cls.terminal_metrics:
            scalar = cls._scalar(cls._find_tag(tags, name))
            if scalar is not None:
                parts.append(f"{name}: {scalar:.4f}")

        return "  ".join(parts)

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ):
        should_log = self.every_n_inner_iters(batch_idx, self.interval)
        should_log_last = (
            self.end_of_epoch(runner.train_dataloader, batch_idx)
            and (
                not self.ignore_last
                or len(runner.train_dataloader) <= self.interval
            )
        )
        if not should_log and not should_log_last:
            return

        # LogProcessor still collects every scalar and the visualizer still
        # writes the complete tag dictionary.  Only the terminal string is
        # reduced, so optimization and offline diagnostics are untouched.
        tags, _ = runner.log_processor.get_log_after_iter(
            runner,
            batch_idx,
            "train",
        )
        runner.logger.info(self._format_train_log(runner, tags))
        runner.visualizer.add_scalars(
            tags,
            step=runner.iter + 1,
            file_path=self.json_log_path,
        )
