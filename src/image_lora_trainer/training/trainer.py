"""Accelerate-based LoRA / QLoRA trainer for Diffusers text-to-image models."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from image_lora_trainer.config import TrainConfig, save_train_config
from image_lora_trainer.data.dataset import build_dataset, collate_batch
from image_lora_trainer.logging_utils import (
    ExperimentTracker,
    format_startup_summary,
    get_logger,
    setup_logging,
)
from image_lora_trainer.models.factory import ModelComponents, load_model
from image_lora_trainer.training.checkpointing import (
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    save_final_adapter,
)
from image_lora_trainer.training.losses import (
    add_noise,
    compute_diffusion_loss,
    compute_time_ids,
    encode_prompt_sd,
    prepare_latents,
    resolve_prediction_type,
)
from image_lora_trainer.training.optimizer import build_optimizer, iter_trainable_parameters
from image_lora_trainer.training.scheduler import build_lr_scheduler
from image_lora_trainer.utils.device import collect_device_info, enable_tf32
from image_lora_trainer.utils.memory import (
    format_oom_message,
    gpu_memory_snapshot,
    reset_peak_memory_stats,
)
from image_lora_trainer.utils.seed import set_seed

logger = get_logger(__name__)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
        "huggingface_hub",
        "safetensors",
        "bitsandbytes",
        "xformers",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


def write_environment_metadata(output_dir: Path, config: TrainConfig) -> None:
    device_info = collect_device_info()
    payload = {
        "model": config.model.pretrained_model_name_or_path,
        "revision": config.model.revision,
        "seed": config.training.seed,
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "device": device_info.to_dict(),
        "method": config.method_name(),
    }
    path = output_dir / "environment.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class DiffusionLoRATrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.output_dir = Path(config.training.output_dir)
        if config.training.run_name:
            self.output_dir = self.output_dir / config.training.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(log_file=self.output_dir / "logs" / "train.log")
        save_train_config(config, self.output_dir / "config.yaml")
        write_environment_metadata(self.output_dir, config)

        if config.training.enable_tf32 or config.training.allow_tf32:
            enable_tf32(True)

        set_seed(config.training.seed)
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=config.training.mixed_precision.value
            if config.training.mixed_precision.value != "no"
            else "no",
            project_config=project_config,
            log_with=None,
        )
        accelerate_set_seed(config.training.seed)

        self.components: ModelComponents | None = None
        self.tracker: ExperimentTracker | None = None
        self.global_step = 0
        self.final_loss: float | None = None
        self.start_time = 0.0

    def train(self) -> dict[str, Any]:
        cfg = self.config
        device_info = collect_device_info()
        logger.info("Device diagnostics: %s", device_info.to_dict())

        dataset = build_dataset(cfg.dataset, cfg.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory and torch.cuda.is_available(),
            collate_fn=collate_batch,
            drop_last=cfg.training.dataloader_drop_last,
        )

        self.components = load_model(cfg)
        components = self.components
        prediction_type = resolve_prediction_type(
            components.scheduler,
            components.architecture,
            cfg.model.prediction_type,
        )

        params = iter_trainable_parameters(
            components.denoiser,
            components.text_encoder,
            components.text_encoder_2,
        )
        optimizer = build_optimizer(cfg.optimizer, params)

        num_update_steps_per_epoch = max(
            1,
            len(dataloader) // cfg.training.gradient_accumulation_steps,
        )
        if cfg.training.max_train_steps is None:
            assert cfg.training.num_train_epochs is not None
            max_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        else:
            max_steps = cfg.training.max_train_steps

        lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_scheduler, max_steps)

        (
            components.denoiser,
            components.text_encoder,
            optimizer,
            dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            components.denoiser,
            components.text_encoder,
            optimizer,
            dataloader,
            lr_scheduler,
        )
        if components.text_encoder_2 is not None:
            components.text_encoder_2 = self.accelerator.prepare(components.text_encoder_2)

        components.vae.to(self.accelerator.device)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        components.vae.to(dtype=weight_dtype)

        resume_path = resolve_resume_path(
            self.output_dir, cfg.training.resume_from_checkpoint
        )
        starting_epoch = 0
        if resume_path is not None:
            state = load_checkpoint(
                resume_path,
                denoiser=self.accelerator.unwrap_model(components.denoiser),
                text_encoder=self.accelerator.unwrap_model(components.text_encoder),
                text_encoder_2=(
                    self.accelerator.unwrap_model(components.text_encoder_2)
                    if components.text_encoder_2 is not None
                    else None
                ),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=self.accelerator,
            )
            self.global_step = int(state.get("global_step", 0))
            starting_epoch = int(state.get("epoch", 0))

        trainable, total = components.trainable_parameter_counts()
        effective_batch = (
            cfg.training.train_batch_size
            * cfg.training.gradient_accumulation_steps
            * max(1, self.accelerator.num_processes)
        )
        summary = format_startup_summary(
            {
                "Model": cfg.model.pretrained_model_name_or_path,
                "Architecture": components.architecture.value,
                "Dataset": cfg.dataset.dataset_path or cfg.dataset.dataset_name,
                "Images": len(dataset),  # type: ignore[arg-type]
                "Resolution": cfg.preprocess.resolution,
                "Training method": cfg.method_name(),
                "Quantization": (
                    f"{cfg.quantization.bits}-bit {cfg.quantization.quant_type}"
                    if cfg.quantization.enabled
                    else "disabled"
                ),
                "Precision": cfg.training.mixed_precision.value,
                "LoRA rank": cfg.lora.rank,
                "Trainable parameters": f"{trainable:,}",
                "Total parameters": f"{total:,}",
                "Trainable percentage": f"{(100 * trainable / total) if total else 0:.4f}%",
                "Batch size": cfg.training.train_batch_size,
                "Gradient accumulation": cfg.training.gradient_accumulation_steps,
                "Effective batch size": effective_batch,
                "Learning rate": cfg.optimizer.learning_rate,
                "Maximum steps": max_steps,
                "Prediction type": prediction_type,
                "Output path": str(self.output_dir),
            }
        )
        logger.info("\n%s", summary)

        if self.accelerator.is_main_process and cfg.logging.backend.value != "none":
            self.tracker = ExperimentTracker(
                backend=cfg.logging.backend.value,
                output_dir=self.output_dir,
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                config=cfg.to_yaml_dict(),
                run_name=cfg.training.run_name,
            )

        reset_peak_memory_stats()
        self.start_time = time.time()
        progress = tqdm(
            total=max_steps,
            initial=self.global_step,
            disable=not self.accelerator.is_local_main_process,
            desc="train",
        )

        num_epochs = cfg.training.num_train_epochs or (
            max_steps // num_update_steps_per_epoch + 1
        )

        try:
            for epoch in range(starting_epoch, num_epochs):
                components.denoiser.train()
                if cfg.lora.resolved_text_encoder().enabled:
                    components.text_encoder.train()
                if (
                    components.text_encoder_2 is not None
                    and cfg.lora.resolved_text_encoder_2().enabled
                ):
                    components.text_encoder_2.train()

                for batch in dataloader:
                    with self.accelerator.accumulate(components.denoiser):
                        loss = self._training_step(
                            batch, components, prediction_type, weight_dtype
                        )
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                params, cfg.optimizer.max_grad_norm
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        progress.update(1)
                        self.final_loss = float(loss.detach().item())
                        self._log_step(loss, lr_scheduler, epoch)

                        if (
                            self.global_step % cfg.training.checkpointing_steps == 0
                            and self.accelerator.is_main_process
                        ):
                            save_checkpoint(
                                output_dir=self.output_dir,
                                global_step=self.global_step,
                                denoiser=self.accelerator.unwrap_model(components.denoiser),
                                text_encoder=self.accelerator.unwrap_model(
                                    components.text_encoder
                                ),
                                text_encoder_2=(
                                    self.accelerator.unwrap_model(components.text_encoder_2)
                                    if components.text_encoder_2 is not None
                                    else None
                                ),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                config_dict=cfg.to_yaml_dict(),
                                accelerator=self.accelerator,
                                epoch=epoch,
                                checkpoints_total_limit=cfg.training.checkpoints_total_limit,
                            )

                        if (
                            cfg.validation.enabled
                            and cfg.validation.prompts
                            and self.global_step % cfg.validation.every_n_steps == 0
                            and self.accelerator.is_main_process
                        ):
                            self._run_validation(components, weight_dtype)

                        if self.global_step >= max_steps:
                            break
                if self.global_step >= max_steps:
                    break
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(format_oom_message(exc))
            raise
        finally:
            progress.close()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_dir = save_final_adapter(
                self.output_dir,
                self.accelerator.unwrap_model(components.denoiser),
                self.accelerator.unwrap_model(components.text_encoder),
                (
                    self.accelerator.unwrap_model(components.text_encoder_2)
                    if components.text_encoder_2 is not None
                    else None
                ),
            )
            summary_payload = self._write_training_summary(final_dir, trainable, total)
            if cfg.hub.push_to_hub:
                self._push_to_hub(final_dir)
        else:
            summary_payload = {}

        if self.tracker is not None:
            self.tracker.close()
        self.accelerator.end_training()
        return summary_payload

    def _training_step(
        self,
        batch: dict[str, Any],
        components: ModelComponents,
        prediction_type: str,
        weight_dtype: torch.dtype,
    ) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(
            device=self.accelerator.device, dtype=weight_dtype
        )

        with torch.no_grad():
            latents = prepare_latents(components.vae, pixel_values, weight_dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            components.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = add_noise(
            components.scheduler,
            latents,
            noise,
            timesteps,
            prediction_type=prediction_type,
        )

        prompt_embeds, pooled = encode_prompt_sd(
            captions=batch["captions"],
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            tokenizer_2=components.tokenizer_2,
            text_encoder_2=components.text_encoder_2,
            device=self.accelerator.device,
            dtype=weight_dtype,
        )

        added_cond_kwargs = None
        if components.capabilities.supports_sdxl_time_ids:
            add_time_ids = compute_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"],
                batch["target_sizes"],
                dtype=weight_dtype,
                device=self.accelerator.device,
            )
            added_cond_kwargs = {
                "text_embeds": pooled,
                "time_ids": add_time_ids,
            }

        if components.is_flux:
            # Minimal FLUX forward: transformer expects packed representations in full
            # recipes. We keep a clear failure if shapes are incompatible rather than
            # silently training incorrectly.
            model_pred = components.denoiser(
                hidden_states=noisy_latents,
                timestep=timesteps / 1000.0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        else:
            unet_kwargs: dict[str, Any] = {
                "sample": noisy_latents,
                "timestep": timesteps,
                "encoder_hidden_states": prompt_embeds,
                "return_dict": False,
            }
            if added_cond_kwargs is not None:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
            model_pred = components.denoiser(**unet_kwargs)[0]

        loss_out = compute_diffusion_loss(
            model_pred=model_pred,
            noise=noise,
            latents=latents,
            timesteps=timesteps,
            scheduler=components.scheduler,
            prediction_type=prediction_type,
        )
        return loss_out.loss

    def _log_step(self, loss: torch.Tensor, lr_scheduler, epoch: int) -> None:
        if self.global_step % self.config.logging.log_every_n_steps != 0:
            return
        if not self.accelerator.is_main_process:
            return
        metrics = {
            "train/loss": float(loss.detach().item()),
            "train/lr": float(lr_scheduler.get_last_lr()[0]),
            "train/epoch": float(epoch),
            "train/step": float(self.global_step),
        }
        snap = gpu_memory_snapshot(self.accelerator.device)
        if snap is not None:
            metrics["system/gpu_allocated_gb"] = snap.allocated_gb
            metrics["system/gpu_peak_gb"] = snap.max_allocated_gb
        if self.tracker is not None:
            self.tracker.log_scalars(metrics, self.global_step)
        logger.info(
            "step=%d loss=%.6f lr=%.3e",
            self.global_step,
            metrics["train/loss"],
            metrics["train/lr"],
        )

    def _run_validation(self, components: ModelComponents, weight_dtype: torch.dtype) -> None:
        from image_lora_trainer.inference.pipeline import generate_validation_images

        out_dir = self.output_dir / "validation" / f"step_{self.global_step:06d}"
        images = generate_validation_images(
            components=components,
            prompts=self.config.validation.prompts,
            output_dir=out_dir,
            seed=self.config.validation.seed,
            num_images_per_prompt=self.config.validation.num_images_per_prompt,
            num_inference_steps=self.config.validation.num_inference_steps,
            guidance_scale=self.config.validation.guidance_scale,
            negative_prompt=self.config.validation.negative_prompt,
            height=self.config.validation.height or self.config.preprocess.resolution,
            width=self.config.validation.width or self.config.preprocess.resolution,
            dtype=weight_dtype,
            accelerator=self.accelerator,
        )
        if self.tracker is not None and images:
            self.tracker.log_images("validation", images[:8], self.global_step)

    def _write_training_summary(
        self, final_dir: Path, trainable: int, total: int
    ) -> dict[str, Any]:
        snap = gpu_memory_snapshot()
        payload = {
            "model": self.config.model.pretrained_model_name_or_path,
            "architecture": (
                self.components.architecture.value if self.components else "unknown"
            ),
            "method": self.config.method_name(),
            "lora_rank": self.config.lora.rank,
            "quantization": (
                {
                    "bits": self.config.quantization.bits,
                    "quant_type": self.config.quantization.quant_type,
                }
                if self.config.quantization.enabled
                else None
            ),
            "steps": self.global_step,
            "training_time_seconds": round(time.time() - self.start_time, 2),
            "final_loss": self.final_loss,
            "peak_gpu_memory_gb": snap.max_allocated_gb if snap else None,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "output_adapter": str(final_dir),
        }
        path = self.output_dir / "training_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote training summary to %s", path)
        return payload

    def _push_to_hub(self, final_dir: Path) -> None:
        if not self.config.hub.repo_id:
            raise ValueError("hub.push_to_hub=true requires hub.repo_id")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            self.config.hub.repo_id,
            private=self.config.hub.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=self.config.hub.repo_id,
            folder_path=str(final_dir),
            commit_message=f"Upload LoRA adapter ({self.config.method_name()})",
        )
        logger.info("Pushed adapters to Hub repo %s", self.config.hub.repo_id)


def run_training(config: TrainConfig) -> dict[str, Any]:
    trainer = DiffusionLoRATrainer(config)
    return trainer.train()
