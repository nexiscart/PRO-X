#!/usr/bin/env python3
"""
Training Manager for Pro Roofing AI
Manages training process, monitoring, and checkpoints
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
import wandb
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datetime import datetime, timedelta

class RoofingTrainingManager:
    """Manages the training process with monitoring and optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.total_steps = 0
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        # Monitoring
        self.loss_history = []
        self.learning_rate_history = []
        self.memory_usage_history = []
        
        # Callbacks
        self.callbacks = []
        
        self.logger.info("🎯 Training Manager initialized")

    def setup_monitoring(self) -> List[TrainerCallback]:
        """Setup training monitoring callbacks"""
        callbacks = []
        
        # Custom progress callback
        callbacks.append(RoofingProgressCallback(self))
        
        # Memory monitoring callback
        if self.config.get('monitoring', {}).get('track_memory', True):
            callbacks.append(MemoryMonitoringCallback(self))
        
        # Performance tracking callback
        if self.config.get('monitoring', {}).get('log_gpu_stats', True):
            callbacks.append(PerformanceCallback(self))
        
        # Early stopping is handled by transformers.EarlyStoppingCallback
        
        self.logger.info(f"✅ Setup {len(callbacks)} monitoring callbacks")
        return callbacks

    def start_training_session(self):
        """Initialize training session"""
        self.training_start_time = datetime.now()
        self.current_epoch = 0
        self.total_steps = 0
        self.best_eval_loss = float('inf')
        
        # Create training log
        self.training_log = {
            "session_id": f"roofing_ai_{int(time.time())}",
            "start_time": self.training_start_time.isoformat(),
            "config": self.config,
            "events": []
        }
        
        self.log_event("training_started", {"message": "Training session initiated"})
        self.logger.info(f"🚀 Training session started: {self.training_log['session_id']}")

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log training event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        self.training_log["events"].append(event)
        
        # Log to file
        log_dir = Path("logs/training")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / f"{self.training_log['session_id']}.log", 'a', encoding='utf-8') as f:
            f.write(f"{event['timestamp']} - {event_type}: {json.dumps(data)}\n")

    def on_epoch_start(self, epoch: int):
        """Handle epoch start"""
        self.current_epoch = epoch
        self.log_event("epoch_started", {"epoch": epoch})
        self.logger.info(f"📍 Starting epoch {epoch}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Handle epoch end"""
        self.log_event("epoch_completed", {
            "epoch": epoch,
            "logs": logs,
            "duration_minutes": (datetime.now() - self.training_start_time).total_seconds() / 60
        })
        
        # Check if this is the best model so far
        eval_loss = logs.get('eval_loss')
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.log_event("new_best_model", {
                "epoch": epoch,
                "eval_loss": eval_loss,
                "improvement": self.best_eval_loss - eval_loss
            })
            self.logger.info(f"🏆 New best model at epoch {epoch}: eval_loss = {eval_loss:.4f}")

    def on_step(self, step: int, logs: Dict[str, float]):
        """Handle training step"""
        self.total_steps = step
        
        # Record metrics
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])
        
        if 'learning_rate' in logs:
            self.learning_rate_history.append(logs['learning_rate'])
        
        # Log memory usage periodically
        if step % 100 == 0:
            memory_stats = self.get_memory_stats()
            self.memory_usage_history.append(memory_stats)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
        
        # CPU memory
        import psutil
        process = psutil.Process()
        stats['cpu_memory'] = process.memory_info().rss / 1024**3
        
        return stats

    def estimate_remaining_time(self, current_step: int, total_steps: int) -> str:
        """Estimate remaining training time"""
        if current_step == 0:
            return "Unknown"
        
        elapsed = datetime.now() - self.training_start_time
        steps_per_second = current_step / elapsed.total_seconds()
        remaining_steps = total_steps - current_step
        remaining_seconds = remaining_steps / steps_per_second
        
        remaining_time = timedelta(seconds=remaining_seconds)
        return str(remaining_time).split('.')[0]  # Remove microseconds

    def save_checkpoint_metadata(self, checkpoint_dir: str, step: int, eval_metrics: Dict[str, float]):
        """Save metadata for checkpoint"""
        metadata = {
            "step": step,
            "epoch": self.current_epoch,
            "timestamp": datetime.now().isoformat(),
            "eval_metrics": eval_metrics,
            "training_duration_minutes": (datetime.now() - self.training_start_time).total_seconds() / 60,
            "memory_stats": self.get_memory_stats(),
            "config_hash": hash(str(self.config))
        }
        
        checkpoint_path = Path(checkpoint_dir)
        metadata_file = checkpoint_path / "checkpoint_metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.log_event("checkpoint_saved", {
            "checkpoint_dir": checkpoint_dir,
            "step": step,
            "eval_metrics": eval_metrics
        })

    def cleanup_old_checkpoints(self, checkpoint_dir: str, keep_best: int = 3):
        """Clean up old checkpoints, keeping only the best ones"""
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            return
        
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        
        if len(checkpoint_dirs) <= keep_best:
            return
        
        # Load metadata for each checkpoint
        checkpoint_info = []
        for ckpt_dir in checkpoint_dirs:
            metadata_file = ckpt_dir / "checkpoint_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    eval_loss = metadata.get('eval_metrics', {}).get('eval_loss', float('inf'))
                    checkpoint_info.append((ckpt_dir, eval_loss, metadata.get('step', 0)))
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata for {ckpt_dir}: {e}")
                    continue
        
        # Sort by eval_loss (ascending) and keep best ones
        checkpoint_info.sort(key=lambda x: x[1])  # Sort by eval_loss
        
        # Remove excess checkpoints
        to_remove = checkpoint_info[keep_best:]
        for ckpt_dir, eval_loss, step in to_remove:
            try:
                import shutil
                shutil.rmtree(ckpt_dir)
                self.logger.info(f"🗑️  Removed checkpoint: {ckpt_dir.name} (eval_loss: {eval_loss:.4f})")
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {ckpt_dir}: {e}")

    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        if not self.training_start_time:
            return {"error": "Training not started"}
        
        total_duration = datetime.now() - self.training_start_time
        
        report = {
            "session_info": {
                "session_id": self.training_log.get('session_id', 'unknown'),
                "start_time": self.training_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_minutes": total_duration.total_seconds() / 60,
                "total_duration_formatted": str(total_duration).split('.')[0]
            },
            "training_progress": {
                "current_epoch": self.current_epoch,
                "total_steps": self.total_steps,
                "best_eval_loss": self.best_eval_loss if self.best_eval_loss != float('inf') else None
            },
            "performance_metrics": {
                "avg_loss": sum(self.loss_history) / len(self.loss_history) if self.loss_history else None,
                "loss_trend": self.calculate_trend(self.loss_history) if len(self.loss_history) > 10 else None,
                "final_learning_rate": self.learning_rate_history[-1] if self.learning_rate_history else None
            },
            "resource_usage": {
                "peak_gpu_memory": max([stats.get('gpu_allocated', 0) for stats in self.memory_usage_history]) if self.memory_usage_history else None,
                "avg_cpu_memory": sum([stats.get('cpu_memory', 0) for stats in self.memory_usage_history]) / len(self.memory_usage_history) if self.memory_usage_history else None
            },
            "training_events": len(self.training_log.get('events', [])),
            "recommendations": self.generate_training_recommendations()
        }
        
        return report

    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope < -0.01:
            return "decreasing"  # Good for loss
        elif slope > 0.01:
            return "increasing"  # Bad for loss
        else:
            return "stable"

    def generate_training_recommendations(self) -> List[str]:
        """Generate recommendations based on training performance"""
        recommendations = []
        
        # Loss trend analysis
        if len(self.loss_history) > 20:
            recent_loss = self.loss_history[-10:]
            early_loss = self.loss_history[10:20]
            
            recent_avg = sum(recent_loss) / len(recent_loss)
            early_avg = sum(early_loss) / len(early_loss)
            
            improvement = (early_avg - recent_avg) / early_avg
            
            if improvement < 0.01:
                recommendations.append("Loss improvement has plateaued. Consider reducing learning rate or early stopping.")
            elif improvement > 0.5:
                recommendations.append("Excellent loss improvement. Current settings are working well.")
        
        # Memory usage analysis
        if self.memory_usage_history:
            peak_gpu = max([stats.get('gpu_allocated', 0) for stats in self.memory_usage_history])
            if peak_gpu > 20:  # > 20GB
                recommendations.append("High GPU memory usage detected. Consider reducing batch size or using gradient checkpointing.")
            elif peak_gpu < 5:  # < 5GB
                recommendations.append("Low GPU memory usage. You might be able to increase batch size for faster training.")
        
        # Training speed analysis
        if self.total_steps > 0 and self.training_start_time:
            elapsed_minutes = (datetime.now() - self.training_start_time).total_seconds() / 60
            steps_per_minute = self.total_steps / elapsed_minutes
            
            if steps_per_minute < 1:
                recommendations.append("Training speed is slow. Consider optimizing data loading or using mixed precision.")
        
        return recommendations

    def save_final_report(self, output_dir: str):
        """Save final training report"""
        report = self.generate_training_report()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        report_file = output_path / f"training_report_{self.training_log.get('session_id', 'unknown')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Save training log
        log_file = output_path / f"training_log_{self.training_log.get('session_id', 'unknown')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2)
        
        self.logger.info(f"📊 Training report saved to: {report_file}")
        self.logger.info(f"📝 Training log saved to: {log_file}")


class RoofingProgressCallback(TrainerCallback):
    """Custom callback for progress tracking"""
    
    def __init__(self, training_manager: RoofingTrainingManager):
        self.training_manager = training_manager

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.training_manager.start_training_session()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.training_manager.on_epoch_start(state.epoch)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            self.training_manager.on_epoch_end(state.epoch, logs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            self.training_manager.on_step(state.global_step, logs)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        eval_metrics = kwargs.get('logs', {})
        self.training_manager.save_checkpoint_metadata(checkpoint_dir, state.global_step, eval_metrics)
        
        # Cleanup old checkpoints
        self.training_manager.cleanup_old_checkpoints(args.output_dir)


class MemoryMonitoringCallback(TrainerCallback):
    """Callback for monitoring memory usage"""
    
    def __init__(self, training_manager: RoofingTrainingManager):
        self.training_manager = training_manager

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % 50 == 0:  # Monitor every 50 steps
            memory_stats = self.training_manager.get_memory_stats()
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "memory/gpu_allocated": memory_stats.get('gpu_allocated', 0),
                    "memory/gpu_reserved": memory_stats.get('gpu_reserved', 0),
                    "memory/cpu_memory": memory_stats.get('cpu_memory', 0)
                }, step=state.global_step)


class PerformanceCallback(TrainerCallback):
    """Callback for performance monitoring"""
    
    def __init__(self, training_manager: RoofingTrainingManager):
        self.training_manager = training_manager
        self.step_start_time = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.step_start_time is not None:
            step_duration = time.time() - self.step_start_time
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "performance/step_duration": step_duration,
                    "performance/steps_per_second": 1.0 / step_duration if step_duration > 0 else 0
                }, step=state.global_step)


def main():
    """Test training manager"""
    config = {
        "monitoring": {
            "track_memory": True,
            "log_gpu_stats": True
        }
    }
    
    manager = RoofingTrainingManager(config)
    
    # Simulate training session
    manager.start_training_session()
    
    # Simulate some training steps
    for epoch in range(2):
        manager.on_epoch_start(epoch)
        
        for step in range(10):
            manager.on_step(step, {"loss": 2.5 - (step * 0.1), "learning_rate": 0.001})
        
        manager.on_epoch_end(epoch, {"eval_loss": 2.0 - (epoch * 0.2)})
    
    # Generate report
    report = manager.generate_training_report()
    print("Training Report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()