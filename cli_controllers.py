"""
CLI控制器模块 - 封装neuralnet.cpp各CLI工具的执行与输出解析

本模块提供以下控制器类：
1. CLIController - 抽象基类，封装进程管理、输出流处理、回调机制
2. MnistTrainController - MNIST训练CLI控制器
3. MnistInferController - MNIST推理CLI控制器  
4. TokenizerTrainController - 分词器训练CLI控制器
5. TokenizerInferController - 分词器推理CLI控制器
6. GptTrainController - GPT训练CLI控制器
7. GptInferController - GPT推理CLI控制器

使用示例：
    controller = MnistTrainController()
    controller.set_output_callback(lambda line: print(f"输出: {line}"))
    controller.set_metric_callback(lambda metric: print(f"指标: {metric}"))
    controller.run(epochs=5, lr=0.001, gpu=True)
"""

import subprocess
import threading
import queue
import re
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RunState(Enum):
    """运行状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Metric:
    """解析出的指标数据"""
    name: str
    value: float
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: Optional[float] = None


class CLIController(ABC):
    """
    CLI控制器抽象基类
    
    提供以下核心功能：
    - 进程管理：启动、停止、监控子进程
    - 输出流处理：实时读取stdout/stderr，通过回调函数逐行通知
    - 参数管理：构建命令行参数
    - 状态管理：跟踪运行状态
    - 错误处理：捕获stderr输出和进程退出码
    """
    
    def __init__(self, executable_path: Optional[str] = None):
        """
        初始化控制器
        
        Args:
            executable_path: 可执行文件路径。如果为None，则使用默认路径（build目录下）
        """
        self.executable_path = executable_path
        self.process: Optional[subprocess.Popen] = None
        self.state = RunState.IDLE
        
        # 回调函数
        self._output_callback: Optional[Callable[[str], None]] = None
        self._metric_callback: Optional[Callable[[Metric], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self._completion_callback: Optional[Callable[[int], None]] = None
        
        # 输出队列和线程
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        
        # 解析状态
        self._collected_metrics: List[Metric] = []
        self._last_output_lines: List[str] = []
        
    @property
    def default_executable_name(self) -> str:
        """返回默认的可执行文件名（不含路径）"""
        # 子类应该重写此方法
        raise NotImplementedError("子类必须实现default_executable_name属性")
    
    def _get_executable_path(self) -> str:
        """获取可执行文件的完整路径"""
        if self.executable_path:
            return self.executable_path
        
        # 尝试在build目录中查找
        build_dir = Path(__file__).parent / "build"
        exe_name = self.default_executable_name
        
        # Windows上需要.exe后缀
        if sys.platform == "win32" and not exe_name.endswith(".exe"):
            exe_name += ".exe"
            
        exe_path = build_dir / exe_name
        if exe_path.exists():
            return str(exe_path)
        
        # 尝试在当前目录查找
        current_dir = Path(__file__).parent
        exe_path = current_dir / exe_name
        if exe_path.exists():
            return str(exe_path)
        
        raise FileNotFoundError(f"找不到可执行文件: {exe_name}")
    
    def set_output_callback(self, callback: Callable[[str], None]):
        """设置输出行回调函数"""
        self._output_callback = callback
        
    def set_metric_callback(self, callback: Callable[[Metric], None]):
        """设置指标解析回调函数"""
        self._metric_callback = callback
        
    def set_error_callback(self, callback: Callable[[str], None]):
        """设置错误输出回调函数"""
        self._error_callback = callback
        
    def set_completion_callback(self, callback: Callable[[int], None]):
        """设置进程完成回调函数（参数为退出码）"""
        self._completion_callback = callback
    
    def _build_command(self, **kwargs) -> List[str]:
        """
        构建命令行参数列表
        
        Args:
            **kwargs: 命令行参数，键值对
            
        Returns:
            命令行参数列表
        """
        cmd = [self._get_executable_path()]
        cmd.extend(self._format_args(**kwargs))
        return cmd
    
    @abstractmethod
    def _format_args(self, **kwargs) -> List[str]:
        """
        格式化参数为命令行参数列表
        
        Args:
            **kwargs: 参数键值对
            
        Returns:
            参数列表
        """
        pass
    
    def _format_arg_value(self, value: Any) -> str:
        """格式化单个参数值为字符串"""
        if isinstance(value, bool):
            return ""  # 布尔参数只作为标志，不带值
        elif isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        else:
            return str(value)
    
    def _should_add_arg(self, key: str, value: Any) -> bool:
        """判断是否应该添加这个参数"""
        # 跳过None值和空字符串
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        return True
    
    def _reader_thread(self, stream, is_error: bool = False):
        """读取流的线程函数"""
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                    
                line = line.rstrip('\n').rstrip('\r')
                if not line:
                    continue
                
                # 添加到输出队列
                self._output_queue.put((line, is_error))
                
                # 如果设置了停止事件，退出线程
                if self._stop_event.is_set():
                    break
                    
        except Exception as e:
            # 将异常作为错误输出
            self._output_queue.put((f"读取线程异常: {e}", True))
        finally:
            try:
                stream.close()
            except:
                pass
    
    def _process_output(self):
        """处理输出队列中的消息"""
        while not self._stop_event.is_set():
            try:
                line, is_error = self._output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if is_error:
                # 错误输出
                if self._error_callback:
                    self._error_callback(line)
                # 也添加到最近输出行（保留最后100行）
                self._last_output_lines.append(f"[stderr] {line}")
                if len(self._last_output_lines) > 100:
                    self._last_output_lines.pop(0)
            else:
                # 正常输出
                if self._output_callback:
                    self._output_callback(line)
                    
                # 尝试解析指标（可能返回单个Metric或列表）
                metric_result = self._parse_metric(line)
                if metric_result:
                    if isinstance(metric_result, list):
                        for metric in metric_result:
                            if self._metric_callback:
                                self._metric_callback(metric)
                            self._collected_metrics.append(metric)
                    elif isinstance(metric_result, Metric):
                        if self._metric_callback:
                            self._metric_callback(metric_result)
                        self._collected_metrics.append(metric_result)
                
                # 保留最近输出行
                self._last_output_lines.append(line)
                if len(self._last_output_lines) > 100:
                    self._last_output_lines.pop(0)
    
    @abstractmethod
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """
        从输出行解析指标
        
        Args:
            line: 输出行
            
        Returns:
            解析出的Metric对象或Metric列表，如果无法解析则返回None
        """
        pass
    
    def run(self, **kwargs) -> int:
        """
        执行CLI命令
        
        Args:
            **kwargs: 命令行参数
            
        Returns:
            进程退出码
            
        Raises:
            RuntimeError: 如果已经在运行
            FileNotFoundError: 如果找不到可执行文件
        """
        if self.state == RunState.RUNNING:
            raise RuntimeError("控制器正在运行中")
        
        # 重置状态
        self.state = RunState.RUNNING
        self._collected_metrics.clear()
        self._last_output_lines.clear()
        self._stop_event.clear()
        
        try:
            cmd = self._build_command(**kwargs)
            
            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动输出读取线程
            self._reader_threads = []
            
            stdout_thread = threading.Thread(
                target=self._reader_thread,
                args=(self.process.stdout, False),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._reader_thread,
                args=(self.process.stderr, True),
                daemon=True
            )
            
            self._reader_threads.extend([stdout_thread, stderr_thread])
            stdout_thread.start()
            stderr_thread.start()
            
            # 启动输出处理线程
            processor_thread = threading.Thread(
                target=self._process_output,
                daemon=True
            )
            processor_thread.start()
            self._reader_threads.append(processor_thread)
            
            # 等待进程完成
            exit_code = self.process.wait()
            
            # 等待所有输出被处理
            time.sleep(0.1)
            
            # 设置状态
            if exit_code == 0:
                self.state = RunState.COMPLETED
            else:
                self.state = RunState.FAILED
            
            # 触发完成回调
            if self._completion_callback:
                self._completion_callback(exit_code)
            
            return exit_code
            
        except Exception as e:
            self.state = RunState.FAILED
            raise
    
    def cancel(self):
        """取消当前运行"""
        if self.state != RunState.RUNNING:
            return
            
        self._stop_event.set()
        self.state = RunState.CANCELLED
        
        if self.process:
            try:
                self.process.terminate()
                # 等待进程终止
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except:
                pass
    
    def get_last_output(self, n: int = 10) -> List[str]:
        """获取最近的n行输出"""
        return self._last_output_lines[-n:]
    
    def get_metrics(self) -> List[Metric]:
        """获取所有解析出的指标"""
        return self._collected_metrics.copy()
    
    def clear_metrics(self):
        """清除所有指标"""
        self._collected_metrics.clear()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.state == RunState.RUNNING


class MnistTrainController(CLIController):
    """
    MNIST训练CLI控制器
    
    封装mnist_train.exe，支持以下功能：
    - 架构选择：MLP或Transformer
    - 训练参数：轮数、学习率、批大小、优化器等
    - 设备选择：CPU、GPU(Vulkan)、CUDA
    - 模型管理：保存、恢复
    - 实时指标：训练损失、验证准确率
    """
    
    @property
    def default_executable_name(self) -> str:
        return "mnist_train"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化MNIST训练参数"""
        args = []
        
        # 架构选择
        if "arch" in kwargs:
            args.extend(["--arch", self._format_arg_value(kwargs["arch"])])
        
        # 数据集路径
        if "dataset" in kwargs:
            args.extend(["--dataset", self._format_arg_value(kwargs["dataset"])])
        
        # 保存路径
        if "save" in kwargs:
            args.extend(["--save", self._format_arg_value(kwargs["save"])])
        
        # 恢复路径
        if "resume" in kwargs:
            args.extend(["--resume", self._format_arg_value(kwargs["resume"])])
        
        # 训练参数
        if "epochs" in kwargs:
            args.extend(["--epochs", self._format_arg_value(kwargs["epochs"])])
        
        if "lr" in kwargs:
            args.extend(["--lr", self._format_arg_value(kwargs["lr"])])
        
        if "batch_size" in kwargs:
            args.extend(["--batch-size", self._format_arg_value(kwargs["batch_size"])])
        
        if "optimizer" in kwargs:
            args.extend(["--optimizer", self._format_arg_value(kwargs["optimizer"])])
        
        if "weight_decay" in kwargs:
            args.extend(["--weight-decay", self._format_arg_value(kwargs["weight_decay"])])
        
        # 设备选择
        if kwargs.get("gpu", False):
            args.append("--gpu")
        
        if kwargs.get("cuda", False):
            args.append("--cuda")
        
        # 其他参数
        if "max_samples" in kwargs:
            args.extend(["--max-samples", self._format_arg_value(kwargs["max_samples"])])
        
        if "shuffle_steps" in kwargs:
            args.extend(["--shuffle-steps", self._format_arg_value(kwargs["shuffle_steps"])])
        
        # MLP专用参数
        if "layer_dims" in kwargs:
            args.extend(["--layer-dims", self._format_arg_value(kwargs["layer_dims"])])

        if "norm" in kwargs:
            args.extend(["--norm", self._format_arg_value(kwargs["norm"])])
        
        # Transformer专用参数
        if "d_model" in kwargs:
            args.extend(["--d-model", self._format_arg_value(kwargs["d_model"])])
        
        if "num_heads" in kwargs:
            args.extend(["--num-heads", self._format_arg_value(kwargs["num_heads"])])
        
        if "num_layers" in kwargs:
            args.extend(["--num-layers", self._format_arg_value(kwargs["num_layers"])])
        
        if "d_ff" in kwargs:
            args.extend(["--d-ff", self._format_arg_value(kwargs["d_ff"])])
        
        if "patch_size" in kwargs:
            args.extend(["--patch-size", self._format_arg_value(kwargs["patch_size"])])
        
        if "eval_samples" in kwargs:
            args.extend(["--eval-samples", self._format_arg_value(kwargs["eval_samples"])])
        
        # 振荡检测
        if "osc_guard" in kwargs:
            args.extend(["--osc-guard", self._format_arg_value(kwargs["osc_guard"])])
        
        if "osc_window" in kwargs:
            args.extend(["--osc-window", self._format_arg_value(kwargs["osc_window"])])
        
        if "osc_threshold" in kwargs:
            args.extend(["--osc-threshold", self._format_arg_value(kwargs["osc_threshold"])])
        
        # 学习率调度
        if "lr_schedule" in kwargs:
            args.extend(["--lr-schedule", self._format_arg_value(kwargs["lr_schedule"])])
        
        if "warmup_epochs" in kwargs:
            args.extend(["--warmup-epochs", self._format_arg_value(kwargs["warmup_epochs"])])
        
        if "min_lr" in kwargs:
            args.extend(["--min-lr", self._format_arg_value(kwargs["min_lr"])])
        
        if "lr_per_epoch" in kwargs:
            args.extend(["--lr-per-epoch", self._format_arg_value(kwargs["lr_per_epoch"])])
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析MNIST训练输出的指标"""
        # 匹配batch进度模式: "Epoch 1/10  batch 1/1  loss: 2.6204  time: 37ms"
        batch_loss_match = re.search(r'Epoch\s+(\d+)/(\d+)\s+batch\s+(\d+)/(\d+)\s+loss:\s*([\d.]+)', line)
        if batch_loss_match:
            epoch = int(batch_loss_match.group(1))
            loss = float(batch_loss_match.group(5))
            return Metric(
                name="batch_loss",
                value=loss,
                epoch=epoch,
                timestamp=time.time()
            )
        
        # 匹配epoch总结模式: "Epoch 1/10  lr=1.0000e-02  loss=2.6204  train_acc=27.00%  test_acc=19.54%  time=0.0s"
        epoch_summary_match = re.search(
            r'Epoch\s+(\d+)/(\d+)\s+lr=([\d.e+-]+)\s+loss=([\d.]+)\s+train_acc=([\d.]+)%\s+test_acc=([\d.]+)%',
            line
        )
        if epoch_summary_match:
            epoch = int(epoch_summary_match.group(1))
            lr = float(epoch_summary_match.group(3))
            avg_loss = float(epoch_summary_match.group(4))
            train_acc = float(epoch_summary_match.group(5))
            test_acc = float(epoch_summary_match.group(6))
            # 返回多个指标
            metrics = []
            metrics.append(Metric(name="learning_rate", value=lr, epoch=epoch, timestamp=time.time()))
            metrics.append(Metric(name="train_loss", value=avg_loss, epoch=epoch, timestamp=time.time()))
            metrics.append(Metric(name="train_accuracy", value=train_acc, epoch=epoch, timestamp=time.time()))
            metrics.append(Metric(name="test_accuracy", value=test_acc, epoch=epoch, timestamp=time.time()))
            return metrics  # 返回列表，由调用方处理
        
        return None


class MnistInferController(CLIController):
    """
    MNIST推理CLI控制器
    
    封装mnist_infer.exe，支持以下功能：
    - 单张图片推理
    - 批量推理
    - GPU/CUDA加速
    - 预测结果展示
    """
    
    @property
    def default_executable_name(self) -> str:
        return "mnist_infer"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化MNIST推理参数"""
        args = []
        
        # 输入文件或目录（第一个位置参数）
        if "input" in kwargs:
            args.append(self._format_arg_value(kwargs["input"]))
        
        # 模型路径
        if "model" in kwargs:
            args.extend(["--model", self._format_arg_value(kwargs["model"])])
        
        # 显示前k个预测
        if "topk" in kwargs:
            args.extend(["--topk", self._format_arg_value(kwargs["topk"])])
        
        # 显示像素
        if kwargs.get("show_pixels", False):
            args.append("--show-pixels")
        
        # 设备选择
        if kwargs.get("gpu", False):
            args.append("--gpu")
        
        if kwargs.get("cuda", False):
            args.append("--cuda")
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析MNIST推理输出的指标"""
        # 匹配预测结果模式: "Prediction: 5 (confidence: 0.98)"
        pred_match = re.search(r'Prediction:\s*(\d+)\s*\(confidence:\s*([\d.]+)\)', line)
        if pred_match:
            digit = int(pred_match.group(1))
            confidence = float(pred_match.group(2))
            return Metric(
                name="prediction",
                value=digit,
                timestamp=time.time()
            )
        
        return None


class TokenizerTrainController(CLIController):
    """
    分词器训练CLI控制器
    
    封装tokenizer_train.exe，支持以下功能：
    - BPE分词器训练
    - 字符级BPE训练
    - 词表大小控制
    - 训练进度监控
    """
    
    @property
    def default_executable_name(self) -> str:
        return "tokenizer_train"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化分词器训练参数"""
        args = []
        
        # 训练文本文件（第一个位置参数）
        if "text_file" in kwargs:
            args.append(self._format_arg_value(kwargs["text_file"]))
        
        # 分词器类型
        if "tokenizer" in kwargs:
            args.extend(["--tokenizer", self._format_arg_value(kwargs["tokenizer"])])
        
        # 输出路径
        if "output" in kwargs:
            args.extend(["--output", self._format_arg_value(kwargs["output"])])
        
        # 词表大小
        if "vocab_size" in kwargs:
            args.extend(["--vocab-size", self._format_arg_value(kwargs["vocab_size"])])
        
        # 最小频率
        if "min_freq" in kwargs:
            args.extend(["--min-freq", self._format_arg_value(kwargs["min_freq"])])
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析分词器训练输出的指标"""
        # 匹配合并操作模式: "Merge 100/5000: 'th' + 'e' → 'the'"
        merge_match = re.search(r'Merge\s+(\d+)/(\d+):', line)
        if merge_match:
            current = int(merge_match.group(1))
            total = int(merge_match.group(2))
            return Metric(
                name="merge_progress",
                value=current / total * 100,  # 百分比
                step=current,
                timestamp=time.time()
            )
        
        # 匹配词表大小模式: "Vocab size: 5000"
        vocab_match = re.search(r'Vocab size:\s*(\d+)', line)
        if vocab_match:
            vocab_size = int(vocab_match.group(1))
            return Metric(
                name="vocab_size",
                value=vocab_size,
                timestamp=time.time()
            )
        
        return None


class TokenizerInferController(CLIController):
    """
    分词器推理CLI控制器
    
    封装tokenizer_infer.exe，支持以下功能：
    - 文本编码
    - 文本解码
    - 交互模式
    - 文件编码
    """
    
    @property
    def default_executable_name(self) -> str:
        return "tokenizer_infer"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化分词器推理参数"""
        args = []
        
        # 词表路径
        if "vocab" in kwargs:
            args.extend(["--vocab", self._format_arg_value(kwargs["vocab"])])
        
        # 编码文本
        if "encode" in kwargs:
            args.extend(["--encode", self._format_arg_value(kwargs["encode"])])
        
        # 解码文本
        if "decode" in kwargs:
            args.extend(["--decode", self._format_arg_value(kwargs["decode"])])
        
        # 编码文件
        if "encode_file" in kwargs:
            args.extend(["--encode-file", self._format_arg_value(kwargs["encode_file"])])
        
        # 交互模式
        if kwargs.get("interactive", False):
            args.append("--interactive")
        
        # 显示字节
        if kwargs.get("show_bytes", False):
            args.append("--show-bytes")
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析分词器推理输出的指标"""
        # 匹配token数量模式: "Tokens: 15"
        tokens_match = re.search(r'Tokens:\s*(\d+)', line)
        if tokens_match:
            token_count = int(tokens_match.group(1))
            return Metric(
                name="token_count",
                value=token_count,
                timestamp=time.time()
            )
        
        return None


class GptTrainController(CLIController):
    """
    GPT训练CLI控制器
    
    封装text_train.exe，支持以下功能：
    - GPT模型训练
    - 多种优化器
    - 学习率调度
    - GPU/CUDA加速
    - TDR防护
    - Checkpoint管理
    """
    
    @property
    def default_executable_name(self) -> str:
        return "text_train"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化GPT训练参数"""
        args = []
        
        # 训练文本文件（第一个位置参数）
        if "text_file" in kwargs:
            args.append(self._format_arg_value(kwargs["text_file"]))
        
        # 保存路径
        if "save" in kwargs:
            args.extend(["--save", self._format_arg_value(kwargs["save"])])
        
        # 恢复路径
        if "resume" in kwargs:
            args.extend(["--resume", self._format_arg_value(kwargs["resume"])])
        
        # 词表路径
        if "vocab" in kwargs:
            args.extend(["--vocab", self._format_arg_value(kwargs["vocab"])])
        
        # 测试集文件
        if "test_file" in kwargs:
            args.extend(["--test-file", self._format_arg_value(kwargs["test_file"])])
        
        # 训练参数
        if "epochs" in kwargs:
            args.extend(["--epochs", self._format_arg_value(kwargs["epochs"])])
        
        if "lr" in kwargs:
            args.extend(["--lr", self._format_arg_value(kwargs["lr"])])
        
        if "batch_size" in kwargs:
            args.extend(["--batch-size", self._format_arg_value(kwargs["batch_size"])])
        
        if "accum_steps" in kwargs:
            args.extend(["--accum-steps", self._format_arg_value(kwargs["accum_steps"])])
        
        if "seq_len" in kwargs:
            args.extend(["--seq-len", self._format_arg_value(kwargs["seq_len"])])
        
        if "stride" in kwargs:
            args.extend(["--stride", self._format_arg_value(kwargs["stride"])])
        
        if "optimizer" in kwargs:
            args.extend(["--optimizer", self._format_arg_value(kwargs["optimizer"])])
        
        if "weight_decay" in kwargs:
            args.extend(["--weight-decay", self._format_arg_value(kwargs["weight_decay"])])
        
        # 模型参数
        if "d_model" in kwargs:
            args.extend(["--d-model", self._format_arg_value(kwargs["d_model"])])
        
        if "num_heads" in kwargs:
            args.extend(["--num-heads", self._format_arg_value(kwargs["num_heads"])])
        
        if "num_layers" in kwargs:
            args.extend(["--num-layers", self._format_arg_value(kwargs["num_layers"])])
        
        if "d_ff" in kwargs:
            args.extend(["--d-ff", self._format_arg_value(kwargs["d_ff"])])
        
        # 设备选择
        if kwargs.get("gpu", False):
            args.append("--gpu")
        
        if kwargs.get("cuda", False):
            args.append("--cuda")
        
        # 位置编码
        if "positional_encoding" in kwargs:
            args.extend(["--positional-encoding", self._format_arg_value(kwargs["positional_encoding"])])
        
        # 激活函数
        if "activation" in kwargs:
            args.extend(["--activation", self._format_arg_value(kwargs["activation"])])
        
        # 归一化
        if "norm" in kwargs:
            args.extend(["--norm", self._format_arg_value(kwargs["norm"])])
        
        # 日志和保存间隔
        if "log_interval" in kwargs:
            args.extend(["--log-interval", self._format_arg_value(kwargs["log_interval"])])
        
        if "save_interval" in kwargs:
            args.extend(["--save-interval", self._format_arg_value(kwargs["save_interval"])])
        
        # 梯度日志
        if kwargs.get("grad_log", False):
            args.append("--grad-log")
        
        # TDR防护
        if "tdr_retry" in kwargs:
            args.extend(["--tdr-retry", self._format_arg_value(kwargs["tdr_retry"])])
        
        if "max_tdr_retries" in kwargs:
            args.extend(["--max-tdr-retries", self._format_arg_value(kwargs["max_tdr_retries"])])
        
        # Batch录制粒度
        if "flush_interval" in kwargs:
            args.extend(["--flush-interval", self._format_arg_value(kwargs["flush_interval"])])
        
        # 梯度检查点（激活重计算 L1）：每 N 个 Transformer block 重算一次
        if "checkpoint_every" in kwargs:
            args.extend(["--checkpoint-every", self._format_arg_value(kwargs["checkpoint_every"])])

        # activation offload（L1-offload）：把激活搬 host-visible（与 checkpoint 互斥）
        if kwargs.get("activation_offload", False):
            args.append("--activation-offload")

        # 学习率调度
        if "lr_schedule" in kwargs:
            args.extend(["--lr-schedule", self._format_arg_value(kwargs["lr_schedule"])])
        
        if "warmup_epochs" in kwargs:
            args.extend(["--warmup-epochs", self._format_arg_value(kwargs["warmup_epochs"])])
        
        if "warmup_steps" in kwargs:
            args.extend(["--warmup-steps", self._format_arg_value(kwargs["warmup_steps"])])
        
        if "min_lr" in kwargs:
            args.extend(["--min-lr", self._format_arg_value(kwargs["min_lr"])])
        
        if "lr_per_epoch" in kwargs:
            args.extend(["--lr-per-epoch", self._format_arg_value(kwargs["lr_per_epoch"])])
        
        if "max_norm" in kwargs:
            args.extend(["--max-norm", self._format_arg_value(kwargs["max_norm"])])
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析GPT训练输出的指标"""
        # 匹配batch进度模式: "Epoch 1/10  step 1/100  loss: 2.3450"
        step_loss_match = re.search(r'Epoch\s+(\d+)/(\d+)\s+step\s+(\d+)/(\d+)\s+loss:\s*([\d.]+)', line)
        if step_loss_match:
            epoch = int(step_loss_match.group(1))
            step = int(step_loss_match.group(3))
            loss = float(step_loss_match.group(5))
            return Metric(
                name="train_loss",
                value=loss,
                step=step,
                epoch=epoch,
                timestamp=time.time()
            )
        
        # 匹配epoch总结模式: "Epoch 1/10  lr=1.0000e-02  avg_loss=2.3450  time=1.2s"
        epoch_summary_match = re.search(
            r'Epoch\s+(\d+)/(\d+)\s+lr=([\d.e+-]+)\s+avg_loss=([\d.]+)',
            line
        )
        if epoch_summary_match:
            epoch = int(epoch_summary_match.group(1))
            lr = float(epoch_summary_match.group(3))
            avg_loss = float(epoch_summary_match.group(4))
            metrics = []
            metrics.append(Metric(name="learning_rate", value=lr, epoch=epoch, timestamp=time.time()))
            metrics.append(Metric(name="train_loss", value=avg_loss, epoch=epoch, timestamp=time.time()))
            return metrics
        
        # 匹配测试损失模式: "  test_loss=2.123"
        test_loss_match = re.search(r'test_loss=([\d.]+)', line)
        if test_loss_match:
            test_loss = float(test_loss_match.group(1))
            return Metric(
                name="test_loss",
                value=test_loss,
                timestamp=time.time()
            )
        
        # 匹配梯度范数模式: "Grad Norm: 1.234" 或 "grad_norm: 1.234"
        grad_match = re.search(r'grad.*norm[=:\s]*([\d.]+)', line, re.IGNORECASE)
        if grad_match:
            grad_norm = float(grad_match.group(1))
            return Metric(
                name="grad_norm",
                value=grad_norm,
                timestamp=time.time()
            )
        
        # 匹配困惑度模式: "Perplexity: 8.56" 或 "perplexity=8.56"
        perplexity_match = re.search(r'perplexity[=:\s]*([\d.]+)', line, re.IGNORECASE)
        if perplexity_match:
            perplexity = float(perplexity_match.group(1))
            return Metric(
                name="perplexity",
                value=perplexity,
                timestamp=time.time()
            )
        
        return None


class GptInferController(CLIController):
    """
    GPT推理CLI控制器
    
    封装text_infer.exe，支持以下功能：
    - 文本生成
    - 交互模式
    - 温度控制
    - GPU/CUDA加速
    """
    
    @property
    def default_executable_name(self) -> str:
        return "text_infer"
    
    def _format_args(self, **kwargs) -> List[str]:
        """格式化GPT推理参数"""
        args = []
        
        # 模型路径
        if "model" in kwargs:
            args.extend(["--model", self._format_arg_value(kwargs["model"])])
        
        # 词表路径
        if "vocab" in kwargs:
            args.extend(["--vocab", self._format_arg_value(kwargs["vocab"])])
        
        # 提示文本
        if "prompt" in kwargs:
            args.extend(["--prompt", self._format_arg_value(kwargs["prompt"])])
        
        # 交互模式
        if kwargs.get("interactive", False):
            args.append("--interactive")
        
        # 最大生成token数
        if "max_tokens" in kwargs:
            args.extend(["--max-tokens", self._format_arg_value(kwargs["max_tokens"])])
        
        # 温度
        if "temperature" in kwargs:
            args.extend(["--temperature", self._format_arg_value(kwargs["temperature"])])
        
        # 设备选择
        if kwargs.get("gpu", False):
            args.append("--gpu")
        
        if kwargs.get("cuda", False):
            args.append("--cuda")
        
        # 显示token
        if kwargs.get("show_tokens", False):
            args.append("--show-tokens")
        
        return args
    
    def _parse_metric(self, line: str) -> Optional[Union[Metric, List[Metric]]]:
        """解析GPT推理输出的指标"""
        # 匹配生成token数模式: "Generated 50 tokens"
        tokens_match = re.search(r'Generated\s+(\d+)\s+tokens', line)
        if tokens_match:
            token_count = int(tokens_match.group(1))
            return Metric(
                name="generated_tokens",
                value=token_count,
                timestamp=time.time()
            )
        
        # 匹配生成时间模式: "Time: 1.23s"
        time_match = re.search(r'Time:\s*([\d.]+)s', line)
        if time_match:
            gen_time = float(time_match.group(1))
            return Metric(
                name="generation_time",
                value=gen_time,
                timestamp=time.time()
            )
        
        # 匹配速度模式: "Speed: 40.5 tokens/s"
        speed_match = re.search(r'Speed:\s*([\d.]+)\s+tokens/s', line)
        if speed_match:
            speed = float(speed_match.group(1))
            return Metric(
                name="generation_speed",
                value=speed,
                timestamp=time.time()
            )
        
        return None


# 测试代码（可选）
if __name__ == "__main__":
    # 简单测试
    def test_output(line):
        print(f"输出: {line}")
    
    def test_metric(metric):
        print(f"指标: {metric.name} = {metric.value}")
    
    def test_error(line):
        print(f"错误: {line}")
    
    # 创建控制器实例
    mnist_train = MnistTrainController()
    mnist_train.set_output_callback(test_output)
    mnist_train.set_metric_callback(test_metric)
    mnist_train.set_error_callback(test_error)
    
    print("控制器模块加载成功")
    print("可用的控制器类:")
    print("  - MnistTrainController")
    print("  - MnistInferController")
    print("  - TokenizerTrainController")
    print("  - TokenizerInferController")
    print("  - GptTrainController")
    print("  - GptInferController")
