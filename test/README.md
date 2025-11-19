# MarkDiffusion 水印算法单元测试

这个目录包含了 MarkDiffusion 项目中所有水印算法的参数化单元测试。

## 📋 目录结构

```
test/
├── test_watermark_algorithms.py  # 主测试文件（参数化测试）
├── pytest.ini                     # Pytest 配置文件
├── README.md                      # 本文档
└── test_method.py                 # 原有的测试文件（保留）
```

## 🎯 支持的水印算法

### 图像水印算法
- **TR** (Tree-Ring)
- **GS** (Gaussian Shading)
- **PRC** (Perceptual Robust Coding)
- **RI** (Robust Invisible)
- **SEAL** (Secure Embedding Algorithm)
- **ROBIN** (Robust Invisible Noise)
- **WIND** (Watermark in Noise Domain)
- **GM** (Generative Model)
- **SFW** (Stable Feature Watermark)

### 视频水印算法
- **VideoShield**
- **VideoMark**

## 🚀 快速开始

### 安装依赖

```bash
pip install pytest pytest-timeout
```

### 运行所有测试

```bash
# 从项目根目录运行
pytest test/test_watermark_algorithms.py -v
```

## 📖 使用方法

### 1. 测试所有算法

```bash
# 测试所有图像水印算法
pytest test/test_watermark_algorithms.py -v -m image

# 测试所有视频水印算法
pytest test/test_watermark_algorithms.py -v -m video
```

### 2. 测试特定算法

```bash
# 测试 TR 算法
pytest test/test_watermark_algorithms.py -v -k "TR"

# 测试 VideoShield 算法
pytest test/test_watermark_algorithms.py -v -k "VideoShield"

# 使用 --algorithm 参数
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

### 3. 测试特定功能

```bash
# 只测试初始化（快速测试）
pytest test/test_watermark_algorithms.py -v -k "initialization"

# 只测试生成功能
pytest test/test_watermark_algorithms.py -v -k "generation"

# 只测试检测功能
pytest test/test_watermark_algorithms.py -v -k "detection"
```

### 4. 跳过耗时测试

```bash
# 跳过生成测试
pytest test/test_watermark_algorithms.py -v --skip-generation

# 跳过检测测试
pytest test/test_watermark_algorithms.py -v --skip-detection

# 只运行快速测试（不包含 slow 标记的测试）
pytest test/test_watermark_algorithms.py -v -m "not slow"
```

### 5. 自定义模型路径

```bash
# 指定图像模型路径
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /path/to/stable-diffusion-model

# 指定视频模型路径
pytest test/test_watermark_algorithms.py -v \
    --video-model-path /path/to/text-to-video-model

# 同时指定两个模型路径
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /path/to/sd-model \
    --video-model-path /path/to/t2v-model
```

## 🧪 测试类型

### 初始化测试
验证水印算法能否正确初始化：
- 加载配置文件
- 创建水印实例
- 验证管道类型

### 生成测试
验证水印算法的生成功能：
- 生成带水印的媒体（图像/视频）
- 生成不带水印的媒体
- 验证输出格式和尺寸

### 检测测试
验证水印算法的检测功能：
- 检测带水印媒体中的水印
- 检测不带水印媒体（负样本）
- 验证检测结果格式

## 📊 测试报告

### 查看详细输出

```bash
# 显示详细的测试输出
pytest test/test_watermark_algorithms.py -v -s

# 显示测试覆盖率
pytest test/test_watermark_algorithms.py -v --cov=watermark
```

### 生成 HTML 报告

```bash
# 安装 pytest-html
pip install pytest-html

# 生成 HTML 报告
pytest test/test_watermark_algorithms.py -v --html=report.html --self-contained-html
```

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithm` | 指定要测试的算法名称 | None (测试所有) |
| `--image-model-path` | 图像生成模型路径 | `stabilityai/stable-diffusion-2-1-base` |
| `--video-model-path` | 视频生成模型路径 | `damo-vilab/text-to-video-ms-1.7b` |
| `--skip-generation` | 跳过生成测试 | False |
| `--skip-detection` | 跳过检测测试 | False |

## 🏷️ 测试标记 (Markers)

| 标记 | 说明 |
|------|------|
| `@pytest.mark.image` | 图像水印测试 |
| `@pytest.mark.video` | 视频水印测试 |
| `@pytest.mark.slow` | 耗时测试（生成和检测） |

使用标记过滤测试：
```bash
# 只运行图像测试
pytest test/test_watermark_algorithms.py -v -m image

# 只运行视频测试
pytest test/test_watermark_algorithms.py -v -m video

# 排除耗时测试
pytest test/test_watermark_algorithms.py -v -m "not slow"
```

## 💡 实用示例

### 示例 1: 快速验证所有算法能否初始化

```bash
pytest test/test_watermark_algorithms.py -v -k "initialization"
```

### 示例 2: 完整测试单个算法

```bash
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

### 示例 3: 测试所有图像算法的生成功能

```bash
pytest test/test_watermark_algorithms.py -v -m image -k "generation"
```

### 示例 4: 在 CI/CD 中运行（跳过耗时测试）

```bash
pytest test/test_watermark_algorithms.py -v \
    -m "not slow" \
    --tb=short \
    --maxfail=3
```

### 示例 5: 调试特定算法的失败

```bash
pytest test/test_watermark_algorithms.py -v \
    --algorithm TR \
    -s \
    --tb=long \
    --pdb
```

## 🔧 故障排除

### 问题 1: 模型加载失败

**错误信息**: `Failed to load image/video model`

**解决方案**:
1. 检查模型路径是否正确
2. 确保有足够的磁盘空间和内存
3. 使用 `--image-model-path` 或 `--video-model-path` 指定本地模型路径

### 问题 2: CUDA 内存不足

**错误信息**: `CUDA out of memory`

**解决方案**:
1. 减少批处理大小
2. 使用 CPU 运行测试（会自动检测）
3. 一次只测试一个算法：`--algorithm TR`

### 问题 3: 测试超时

**错误信息**: `Test timeout`

**解决方案**:
1. 增加超时时间：在 `pytest.ini` 中修改 `timeout` 值
2. 跳过耗时测试：`--skip-generation --skip-detection`
3. 只运行初始化测试：`-k "initialization"`

### 问题 4: 配置文件未找到

**错误信息**: `Config file not found`

**解决方案**:
1. 确保从项目根目录运行测试
2. 检查 `config/` 目录中是否存在对应的 JSON 配置文件
3. 验证配置文件名称大小写是否正确

## 📝 添加新的测试

如果你想为新的水印算法添加测试，只需：

1. 在 `watermark/auto_watermark.py` 中注册新算法
2. 在 `config/` 目录中添加配置文件
3. 测试框架会自动发现并测试新算法

不需要修改测试代码！

## 🤝 贡献

如果你发现测试中的问题或想要改进测试框架，请：

1. 创建 Issue 描述问题
2. 提交 Pull Request 并附上测试结果
3. 确保所有现有测试仍然通过

## 📄 许可证

本测试代码遵循 MarkDiffusion 项目的 Apache 2.0 许可证。
