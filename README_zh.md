<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# 潜在扩散模型生成式水印的开源工具包

[![Homepage](https://img.shields.io/badge/Homepage-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![HF Models](https://img.shields.io/badge/HF--Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 

**语言版本:** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md)

</div>

> 🔥 **作为一个新发布的项目，我们欢迎 PR！** 如果您已经实现了 LDM 水印算法或有兴趣贡献一个算法，我们很乐意将其包含在 MarkDiffusion 中。加入我们的社区，帮助让生成式水印技术对每个人都更易用！

## 目录
- [注意事项](#-注意事项)
- [更新日志](#-更新日志)
- [MarkDiffusion 简介](#markdiffusion-简介)
  - [概述](#概述)
  - [核心特性](#核心特性)
  - [已实现算法](#已实现算法)
  - [评估模块](#评估模块)
- [安装](#安装)
- [快速开始](#快速开始)
- [如何使用工具包](#如何使用工具包)
  - [生成和检测水印媒体](#生成和检测水印媒体)
  - [可视化水印机制](#可视化水印机制)
  - [评估流水线](#评估流水线)
- [引用](#引用)

## ❗❗❗ 注意事项
随着 MarkDiffusion 仓库内容日益丰富且体积不断增大，我们在 Hugging Face 上创建了一个名为 [Generative-Watermark-Toolkits](https://huggingface.co/Generative-Watermark-Toolkits) 的模型存储仓库以便于使用。该仓库包含了各种涉及自训练模型的水印算法的默认模型。我们已从主仓库中这些水印算法对应的 `ckpts/` 文件夹中移除了模型权重。**使用代码时，请首先根据配置路径从 Hugging Face 仓库下载相应的模型，并将其保存到 `ckpts/` 目录后再运行代码。**

## 🔥 更新日志
🎯 **(2025.10.10)** 添加 *Mask、Overlay、AdaptiveNoiseInjection* 图像攻击工具，感谢付哲语的 PR！

🎯 **(2025.10.09)** 添加 *VideoCodecAttack、FrameRateAdapter、FrameInterpolationAttack* 视频攻击工具，感谢司璐阳的 PR！

🎯 **(2025.10.08)** 添加 *SSIM、BRISQUE、VIF、FSIM* 图像质量分析器，感谢王欢的 PR！

✨ **(2025.10.07)** 添加 [SFW](https://arxiv.org/pdf/2509.07647) 水印方法，感谢王欢的 PR！

✨ **(2025.10.07)** 添加 [VideoMark](https://arxiv.org/abs/2504.16359) 水印方法，感谢李瀚谦的 PR！

✨ **(2025.9.29)** 添加 [GaussMarker](https://arxiv.org/abs/2506.11444) 水印方法，感谢司璐阳的 PR！

## MarkDiffusion 简介

### 概述

MarkDiffusion 是一个用于潜在扩散模型生成式水印的开源 Python 工具包。随着基于扩散的生成模型应用范围的扩大，确保生成媒体的真实性和来源变得至关重要。MarkDiffusion 简化了水印技术的访问、理解和评估，使研究人员和更广泛的社区都能轻松使用。*注意：如果您对 LLM 水印（文本水印）感兴趣，请参考我们团队的 [MarkLLM](https://github.com/THU-BPM/MarkLLM) 工具包。*

该工具包包含三个关键组件：统一的实现框架，用于简化水印算法集成和用户友好的界面；机制可视化套件，直观地展示添加和提取的水印模式，帮助公众理解；以及全面的评估模块，提供 24 个工具的标准实现，涵盖三个关键方面——可检测性、鲁棒性和输出质量，以及 8 个自动化评估流水线。

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### 核心特性

- **统一实现框架：** MarkDiffusion 提供了一个模块化架构，支持八种最先进的 LDM 生成式图像/视频水印算法。

- **全面的算法支持：** 目前实现了来自两大类别的 8 种水印算法：基于模式的方法（Tree-Ring、Ring-ID、ROBIN、WIND）和基于密钥的方法（Gaussian-Shading、PRC、SEAL、VideoShield）。

- **可视化解决方案：** 该工具包包含定制的可视化工具，能够清晰而深入地展示不同水印算法在各种场景下的运行方式。这些可视化有助于揭示算法机制，使其对用户更易理解。

- **评估模块：** 拥有 20 个评估工具，涵盖可检测性、鲁棒性和对输出质量的影响，MarkDiffusion 提供全面的评估能力。它具有 5 个自动化评估流水线：水印检测流水线、图像质量分析流水线、视频质量分析流水线以及专门的鲁棒性评估工具。

### 已实现算法

| **算法** | **类别** | **目标** | **参考文献** |
|---------------|-------------|------------|---------------|
| Tree-Ring | 模式 | 图像 | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | 模式 | 图像 | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | 模式 | 图像 | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | 模式 | 图像 | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | 模式 | 图像 | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | 密钥 | 图像 | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | 密钥 | 图像 | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | 密钥 | 图像 | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | 密钥 | 图像 | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | 密钥 | 视频 | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | 密钥 | 视频 | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### 评估模块
#### 评估流水线

MarkDiffusion 支持八个流水线，两个用于检测（WatermarkedMediaDetectionPipeline 和 UnWatermarkedMediaDetectionPipeline），六个用于质量分析。下表详细说明了质量分析流水线。

| **质量分析流水线** | **输入类型** | **所需数据** | **适用指标** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | 单张图像 | 生成的有/无水印图像 | 单张图像评估指标 | 
| ReferencedImageQualityAnalysisPipeline | 图像 + 参考内容 | 生成的有/无水印图像 + 参考图像/文本 | 需要在单张图像和参考内容（文本/图像）之间计算的指标 | 
| GroupImageQualityAnalysisPipeline | 图像集（+ 参考图像集） | 生成的有/无水印图像集（+ 参考图像集） | 需要在图像集上计算的指标 | 
| RepeatImageQualityAnalysisPipeline | 图像集 | 重复生成的有/无水印图像集 | 用于评估重复生成图像集的指标 | 
| ComparedImageQualityAnalysisPipeline | 两张对比图像 | 生成的有水印和无水印图像 | 测量两张图像之间差异的指标 | 
| DirectVideoQualityAnalysisPipeline | 单个视频 | 生成的视频帧集 | 整体视频评估指标 |

#### 评估工具

| **工具名称** | **评估类别** | **功能描述** | **输出指标** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | 可检测性 | 计算固定阈值水印检测的分类指标 | 各种分类指标 |
| DynamicThresholdSuccessRateCalculator | 可检测性 | 计算动态阈值水印检测的分类指标 | 各种分类指标 |
| **图像攻击工具** | | | |
| Rotation | 鲁棒性（图像） | 图像旋转攻击，测试水印对旋转变换的抗性 | 旋转后的图像/帧 |
| CrSc（裁剪与缩放） | 鲁棒性（图像） | 裁剪和缩放攻击，评估水印对尺寸变化的鲁棒性 | 裁剪/缩放后的图像/帧 |
| GaussianNoise | 鲁棒性（图像） | 高斯噪声攻击，测试水印对噪声干扰的抗性 | 噪声损坏的图像/帧 |
| GaussianBlurring | 鲁棒性（图像） | 高斯模糊攻击，评估水印对模糊处理的抗性 | 模糊后的图像/帧 |
| JPEGCompression | 鲁棒性（图像） | JPEG 压缩攻击，测试水印对有损压缩的鲁棒性 | 压缩后的图像/帧 |
| Brightness | 鲁棒性（图像） | 亮度调整攻击，评估水印对亮度变化的抗性 | 亮度修改后的图像/帧 |
| Mask | 鲁棒性（图像） | 图像遮罩攻击，测试水印对随机黑色矩形部分遮挡的抗性 | 遮罩后的图像/帧 |
| Overlay | 鲁棒性（图像） | 图像覆盖攻击，测试水印对涂鸦式笔触和注释的抗性 | 覆盖后的图像/帧 |
| AdaptiveNoiseInjection | 鲁棒性（图像） | 自适应噪声注入攻击，测试水印对内容感知噪声的抗性（高斯/椒盐/泊松/斑点） | 自适应噪声处理后的图像/帧 |
| **视频攻击工具** | | | |
| MPEG4Compression | 鲁棒性（视频） | MPEG-4 视频压缩攻击，测试视频水印的压缩鲁棒性 | 压缩后的视频帧 |
| FrameAverage | 鲁棒性（视频） | 帧平均攻击，通过帧间平均破坏水印 | 平均后的视频帧 |
| FrameSwap | 鲁棒性（视频） | 帧交换攻击，通过改变帧序列测试鲁棒性 | 交换后的视频帧 |
| VideoCodecAttack | 鲁棒性（视频） | 编解码器重编码攻击，模拟平台转码（H.264/H.265/VP9/AV1） | 重编码后的视频帧 |
| FrameRateAdapter | 鲁棒性（视频） | 帧率转换攻击，在保持时长的同时重采样帧 | 重采样后的帧序列 |
| FrameInterpolationAttack | 鲁棒性（视频） | 帧插值攻击，插入混合帧以改变时间密度 | 插值后的视频帧 |
| **图像质量分析器** | | | |
| InceptionScoreCalculator | 质量（图像） | 评估生成图像的质量和多样性 | IS 分数 |
| FIDCalculator | 质量（图像） | Fréchet Inception Distance，测量生成图像和真实图像之间的分布差异 | FID 值 |
| LPIPSAnalyzer | 质量（图像） | 学习感知图像块相似度，评估感知质量 | LPIPS 距离 |
| CLIPScoreCalculator | 质量（图像） | 基于 CLIP 的文本-图像一致性评估 | CLIP 相似度分数 |
| PSNRAnalyzer | 质量（图像） | 峰值信噪比，测量图像失真 | PSNR 值（dB） |
| NIQECalculator | 质量（图像） | 自然图像质量评估器，无参考质量评估 | NIQE 分数 |
| SSIMAnalyzer | 质量（图像） | 两张图像之间的结构相似性指数 | SSIM 值 |
| BRISQUEAnalyzer | 质量（图像） | 盲/无参考图像空间质量评估器，无需参考即可评估图像的感知质量 | BRISQUE 分数 |
| VIFAnalyzer | 质量（图像） | 视觉信息保真度分析器，比较失真图像与参考图像以量化保留的视觉信息量 | VIF 值 |
| FSIMAnalyzer | 质量（图像） | 特征相似性指数分析器，基于相位一致性和梯度幅度比较两张图像的结构相似性 | FSIM 值 |
| **视频质量分析器** | | | |
| SubjectConsistencyAnalyzer | 质量（视频） | 评估视频中主体对象的一致性 | 主体一致性分数 |
| BackgroundConsistencyAnalyzer | 质量（视频） | 评估视频中背景的连贯性和稳定性 | 背景一致性分数 |
| MotionSmoothnessAnalyzer | 质量（视频） | 评估视频运动的平滑度 | 运动平滑度指标 |
| DynamicDegreeAnalyzer | 质量（视频） | 测量视频中的动态水平和变化幅度 | 动态度值 |
| ImagingQualityAnalyzer | 质量（视频） | 综合评估视频成像质量 | 成像质量分数 |

## 安装

### 环境设置

- Python 3.10+
- PyTorch
- 安装依赖：

```bash
pip install -r requirements.txt
```

*注意：* 某些算法可能需要额外的设置步骤。请参考各个算法文档了解具体要求。

## 快速开始

这里有一个简单的示例帮助您开始使用 MarkDiffusion：

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# 设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 配置扩散流水线
scheduler = DPMSolverMultistepScheduler.from_pretrained("model_path", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("model_path", scheduler=scheduler).to(device)
diffusion_config = DiffusionConfig(
    scheduler=scheduler,
    pipe=pipe,
    device=device,
    image_size=(512, 512),
    num_inference_steps=50,
    guidance_scale=7.5,
    gen_seed=42,
    inversion_type="ddim"
)

# 加载水印算法
watermark = AutoWatermark.load('TR', 
                              algorithm_config='config/TR.json',
                              diffusion_config=diffusion_config)

# 生成带水印的媒体
prompt = "A beautiful sunset over the ocean"
watermarked_image = watermark.generate_watermarked_media(prompt)

# 检测水印
detection_result = watermark.detect_watermark_in_media(watermarked_image)
print(f"Watermark detected: {detection_result}")
```

## 如何使用工具包

我们在 `MarkDiffusion_demo.ipynb` 中提供了大量示例。

### 生成和检测水印媒体

#### 生成和检测水印媒体的案例

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig

# 加载水印算法
mywatermark = AutoWatermark.load(
    'GS',
    algorithm_config=f'config/GS.json',
    diffusion_config=diffusion_config
)

# 生成带水印的图像
watermarked_image = mywatermark.generate_watermarked_media(
    input_data="A beautiful landscape with a river and mountains"
)

# 可视化带水印的图像
watermarked_image.show()

# 检测水印
detection_result = mywatermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

### 可视化水印机制

该工具包包含定制的可视化工具，能够清晰而深入地展示不同水印算法在各种场景下的运行方式。这些可视化有助于揭示算法机制，使其对用户更易理解。

<img src="img/fig2_visualization_mechanism.png" alt="Watermarking Mechanism Visualization" style="zoom:40%;" />

#### 可视化水印机制的案例

```python
from visualize.auto_visualization import AutoVisualizer

# 获取用于可视化的数据
data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)

# 加载可视化器
visualizer = AutoVisualizer.load('GS', 
                                data_for_visualization=data_for_visualization)

# 在 Matplotlib 画布上绘制图表
fig = visualizer.visualize(rows=2, cols=2, 
                          methods=['draw_watermark_bits', 
                                  'draw_reconstructed_watermark_bits', 
                                  'draw_inverted_latents', 
                                  'draw_inverted_latents_fft'])
```

### 评估流水线

#### 评估案例

1. **水印检测流水线**

```python
from evaluation.dataset import StableDiffusionPromptsDataset
from evaluation.pipelines.detection import (
    WatermarkedMediaDetectionPipeline, 
    UnWatermarkedMediaDetectionPipeline, 
    DetectionPipelineReturnType
)
from evaluation.tools.image_editor import JPEGCompression
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator

# 数据集
my_dataset = StableDiffusionPromptsDataset(max_samples=200)

# 设置检测流水线
pipeline1 = WatermarkedMediaDetectionPipeline(
    dataset=my_dataset,
    media_editor_list=[JPEGCompression(quality=60)],
    show_progress=True, 
    return_type=DetectionPipelineReturnType.SCORES
)

pipeline2 = UnWatermarkedMediaDetectionPipeline(
    dataset=my_dataset,
    media_editor_list=[],
    show_progress=True, 
    return_type=DetectionPipelineReturnType.SCORES
)

# 配置检测参数
detection_kwargs = {
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
}

# 计算成功率
calculator = DynamicThresholdSuccessRateCalculator(
    labels=labels, 
    rule=rules,
    target_fpr=target_fpr
)

results = calculator.calculate(
    pipeline1.evaluate(my_watermark, detection_kwargs=detection_kwargs),
    pipeline2.evaluate(my_watermark, detection_kwargs=detection_kwargs)
)
print(results)
```

2. **图像质量分析流水线**

```python
from evaluation.dataset import StableDiffusionPromptsDataset, MSCOCODataset
from evaluation.pipelines.image_quality_analysis import (
    DirectImageQualityAnalysisPipeline,
    ReferencedImageQualityAnalysisPipeline,
    GroupImageQualityAnalysisPipeline,
    RepeatImageQualityAnalysisPipeline,
    ComparedImageQualityAnalysisPipeline,
    QualityPipelineReturnType
)
from evaluation.tools.image_quality_analyzer import (
    NIQECalculator, CLIPScoreCalculator, FIDCalculator, 
    InceptionScoreCalculator, LPIPSAnalyzer, PSNRAnalyzer
)

# 不同质量指标的示例：

# NIQE（无参考图像质量评估器）
if metric == 'NIQE':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[NIQECalculator()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# CLIP 分数
elif metric == 'CLIP':
    my_dataset = MSCOCODataset(max_samples=max_samples)
    pipeline = ReferencedImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[CLIPScoreCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# FID（Fréchet Inception Distance）
elif metric == 'FID':
    my_dataset = MSCOCODataset(max_samples=max_samples)
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[FIDCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# IS（Inception Score）
elif metric == 'IS':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[InceptionScoreCalculator()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# LPIPS（学习感知图像块相似度）
elif metric == 'LPIPS':
    my_dataset = StableDiffusionPromptsDataset(max_samples=10)
    pipeline = RepeatImageQualityAnalysisPipeline(
        dataset=my_dataset,
        prompt_per_image=20,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[LPIPSAnalyzer()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# PSNR（峰值信噪比）
elif metric == 'PSNR':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = ComparedImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[PSNRAnalyzer()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# 加载水印并评估
my_watermark = AutoWatermark.load(
    f'{algorithm_name}',
    algorithm_config=f'config/{algorithm_name}.json',
    diffusion_config=diffusion_config
)

print(pipeline.evaluate(my_watermark))
```

3. **视频质量分析流水线**

```python
from evaluation.dataset import VBenchDataset
from evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline
from evaluation.tools.video_quality_analyzer import (
    SubjectConsistencyAnalyzer,
    MotionSmoothnessAnalyzer,
    DynamicDegreeAnalyzer,
    BackgroundConsistencyAnalyzer,
    ImagingQualityAnalyzer
)

# 加载 VBench 数据集
my_dataset = VBenchDataset(max_samples=200, dimension=dimension)

# 根据指标初始化分析器
if metric == 'subject_consistency':
    analyzer = SubjectConsistencyAnalyzer(device=device)
elif metric == 'motion_smoothness':
    analyzer = MotionSmoothnessAnalyzer(device=device)
elif metric == 'dynamic_degree':
    analyzer = DynamicDegreeAnalyzer(device=device)
elif metric == 'background_consistency':
    analyzer = BackgroundConsistencyAnalyzer(device=device)
elif metric == 'imaging_quality':
    analyzer = ImagingQualityAnalyzer(device=device)
else:
    raise ValueError(f'Invalid metric: {metric}. Supported metrics: 
                    subject_consistency, motion_smoothness, dynamic_degree,
                    background_consistency, imaging_quality')

# 创建视频质量分析流水线
pipeline = DirectVideoQualityAnalysisPipeline(
    dataset=my_dataset,
    watermarked_video_editor_list=[],
    unwatermarked_video_editor_list=[],
    watermarked_frame_editor_list=[],
    unwatermarked_frame_editor_list=[],
    analyzers=[analyzer],
    show_progress=True,
    return_type=QualityPipelineReturnType.MEAN_SCORES
)

print(pipeline.evaluate(my_watermark))
```

## 引用
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```

