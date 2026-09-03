from mmengine.hooks import Hook
from mmengine.runner import Runner
import os.path as osp
import os


class BestModelHook(Hook):
    """Custom hook to save only the single best model for val and test."""
    priority = 'NORMAL'

    def __init__(self, val_metric='mDice', test_metric='mDice'):
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.best_val_score = -1
        self.best_test_score = -1
        self.val_ckpt_path = None
        self.test_ckpt_path = None

    def after_val_epoch(self, runner: Runner) -> None:
        if not self.every_n_iters(runner, 1000): return

        metrics = runner.message_hub.get_info('val')
        if self.val_metric in metrics:
            score = metrics[self.val_metric]
            if score > self.best_val_score:
                self.best_val_score = score
                if self.val_ckpt_path and osp.exists(self.val_ckpt_path):
                    os.remove(self.val_ckpt_path)

                self.val_ckpt_path = osp.join(runner.work_dir, f'best_val_{score:.4f}.pth')
                runner.save_checkpoint(runner.work_dir, osp.basename(self.val_ckpt_path),
                                       save_optimizer=False, save_param_scheduler=False)
                runner.logger.info(f'New best val model: {score:.4f}')

    def after_test_epoch(self, runner: Runner) -> None:
        if not self.every_n_iters(runner, 1000): return

        metrics = runner.message_hub.get_info('test')
        if self.test_metric in metrics:
            score = metrics[self.test_metric]
            if score > self.best_test_score:
                self.best_test_score = score
                if self.test_ckpt_path and osp.exists(self.test_ckpt_path):
                    os.remove(self.test_ckpt_path)

                self.test_ckpt_path = osp.join(runner.work_dir, f'best_test_{score:.4f}.pth')
                runner.save_checkpoint(runner.work_dir, osp.basename(self.test_ckpt_path),
                                       save_optimizer=False, save_param_scheduler=False)
                runner.logger.info(f'New best test model: {score:.4f}')
