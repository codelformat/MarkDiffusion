# MarkDiffusion 水印算法单元测试

这个目录包含了 MarkDiffusion 项目中所有水印算法和反演模块的参数化单元测试。

## 📋 目录结构

```
test/
├── test_watermark_algorithms.py  # 主测试文件（参数化测试）
├── conftest.py                   # Pytest 配置和 fixtures
├── pytest.ini                    # Pytest 配置文件
├── requirements-test.txt         # 测试依赖包
├── run_tests.sh                  # 便捷测试脚本
├── README.md                     # 本文档
└── test_method.py                # 原有的测试文件（保留）
```

## 🎯 支持的测试对象

### 水印算法

#### 图像水印算法（9个）
- **TR** (Tree-Ring)
- **GS** (Gaussian Shading)
- **PRC** (Perceptual Robust Coding)
- **RI** (Robust Invisible)
- **SEAL** (Secure Embedding Algorithm)
- **ROBIN** (Robust Invisible Noise)
- **WIND** (Watermark in Noise Domain)
- **GM** (Generative Model / GaussMarker)
- **SFW** (Stable Feature Watermark)

#### 视频水印算法（2个）
- **VideoShield**
- **VideoMark**

### 反演模块（Inversion Modules）

- **DDIM Inversion** - 支持4D图像输入和5D视频输入
- **Exact Inversion** - 支持4D图像输入

## 🚀 快速开始

### 1. 安装测试依赖

```bash
pip install -r test/requirements-test.txt
```

测试依赖包括：
- pytest
- pytest-timeout
- pytest-html (可选，用于生成HTML报告)
- pytest-cov (可选，用于覆盖率报告)
- pytest-xdist (可选，用于并行测试)

### 2. 运行测试

#### 使用 pytest 直接运行

```bash
# 测试所有算法和模块
pytest test/test_watermark_algorithms.py -v

# 测试特定算法
pytest test/test_watermark_algorithms.py -v --algorithm TR

# 快速测试（仅初始化）
pytest test/test_watermark_algorithms.py -v -k initialization
```

#### 使用便捷脚本

```bash
# 测试所有算法
./test/run_tests.sh

# 测试图像算法
./test/run_tests.sh --type image

# 测试特定算法
./test/run_tests.sh --algorithm TR

# 快速测试（仅初始化）
./test/run_tests.sh --type quick
```

## 📋 测试类型和覆盖范围

### 水印算法测试

#### 1. 初始化测试（11个测试）
验证水印算法能否正确初始化：
- 加载配置文件
- 创建水印实例
- 验证管道类型

```bash
pytest test/test_watermark_algorithms.py -v -k initialization
```

#### 2. 生成测试（22个测试）
验证水印算法的生成功能：
- 生成带水印的媒体（图像/视频）
- 生成不带水印的媒体
- 验证输出格式和尺寸

```bash
# 测试所有生成功能
pytest test/test_watermark_algorithms.py -v -k generation

# 跳过生成测试
pytest test/test_watermark_algorithms.py -v --skip-generation
```

#### 3. 检测测试（11个测试）
验证水印算法的检测功能：
- 检测带水印媒体中的水印
- 检测不带水印媒体（负样本）
- 验证检测结果格式

```bash
# 测试所有检测功能
pytest test/test_watermark_algorithms.py -v -k detection

# 跳过检测测试
pytest test/test_watermark_algorithms.py -v --skip-detection
```

### 反演模块测试（Inversion Tests）

#### 4. 4D图像反演测试（2个测试：DDIM + Exact）
测试反演模块处理4维图像输入的能力：
- 输入形状：`(batch_size, channels, height, width)`
- 测试DDIM和Exact两种反演方法
- 验证能够准确还原潜在向量Z_T

```bash
# 测试4D图像反演
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d"

# 测试DDIM反演
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d[ddim]"

# 测试Exact反演
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d[exact]"
```

#### 5. 5D视频反演测试（1个测试：DDIM）
测试反演模块处理5维视频帧输入的能力：
- 输入形状：`(batch_size, num_frames, channels, height, width)`
- 测试DDIM反演方法
- 验证能够准确还原视频帧的潜在向量Z_T

```bash
# 测试5D视频反演
pytest test/test_watermark_algorithms.py -v -k "test_inversion_5d"
```

#### 6. 反演重建精度测试（1个测试）
测试反演模块的重建精度：
- 前向扩散：x_0 → x_T
- 反向扩散：x_T → x_0
- 验证重建误差在可接受范围内

```bash
# 测试重建精度
pytest test/test_watermark_algorithms.py -v -k "test_inversion_reconstruction"
```

#### 反演测试汇总

```bash
# 测试所有反演模块
pytest test/test_watermark_algorithms.py -v -m inversion

# 测试反演模块（不包括耗时的视频测试）
pytest test/test_watermark_algorithms.py -v -m "inversion and not slow"
```

**总计**: 47+ 个参数化测试用例（44个水印算法测试 + 4个反演测试）

## 📖 常用命令速查表

| 需求 | 命令 |
|------|------|
| 测试所有算法 | `pytest test/test_watermark_algorithms.py -v` |
| 测试图像算法 | `pytest test/test_watermark_algorithms.py -v -m image` |
| 测试视频算法 | `pytest test/test_watermark_algorithms.py -v -m video` |
| 测试反演模块 | `pytest test/test_watermark_algorithms.py -v -m inversion` |
| 测试TR算法 | `pytest test/test_watermark_algorithms.py -v -k TR` |
| 快速测试（初始化） | `pytest test/test_watermark_algorithms.py -v -k initialization` |
| 跳过生成测试 | `pytest test/test_watermark_algorithms.py -v --skip-generation` |
| 并行运行 | `pytest test/test_watermark_algorithms.py -v -n auto` |
| 生成HTML报告 | `pytest test/test_watermark_algorithms.py -v --html=report.html` |
| 测试4D图像反演 | `pytest test/test_watermark_algorithms.py -v -k test_inversion_4d` |
| 测试5D视频反演 | `pytest test/test_watermark_algorithms.py -v -k test_inversion_5d` |

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algorithm` | 指定要测试的算法名称 | None (测试所有) |
| `--image-model-path` | 图像生成模型路径 | `stabilityai/stable-diffusion-2-1-base` |
| `--video-model-path` | 视频生成模型路径 | `damo-vilab/text-to-video-ms-1.7b` |
| `--skip-generation` | 跳过生成测试 | False |
| `--skip-detection` | 跳过检测测试 | False |

## 🏷️ 测试标记 (Markers)

| 标记 | 说明 | 使用方法 |
|------|------|---------|
| `@pytest.mark.image` | 图像水印测试 | `-m image` |
| `@pytest.mark.video` | 视频水印测试 | `-m video` |
| `@pytest.mark.inversion` | 反演模块测试 | `-m inversion` |
| `@pytest.mark.slow` | 耗时测试（生成和检测） | `-m "not slow"` |

使用标记过滤测试：
```bash
# 只运行图像测试
pytest test/test_watermark_algorithms.py -v -m image

# 只运行视频测试
pytest test/test_watermark_algorithms.py -v -m video

# 只运行反演测试
pytest test/test_watermark_algorithms.py -v -m inversion

# 排除耗时测试
pytest test/test_watermark_algorithms.py -v -m "not slow"

# 组合标记：测试图像算法的初始化
pytest test/test_watermark_algorithms.py -v -m image -k initialization
```

## 💡 实用示例

### 示例 1: 快速验证所有算法能否初始化

```bash
pytest test/test_watermark_algorithms.py -v -k "initialization"
```
**预期结果**: 11个算法测试通过，耗时10-30秒

### 示例 2: 完整测试单个算法

```bash
pytest test/test_watermark_algorithms.py -v --algorithm TR
```
**预期结果**: 3个测试通过（初始化、生成、检测）

### 示例 3: 测试所有图像算法的生成功能

```bash
pytest test/test_watermark_algorithms.py -v -m image -k "generation"
```
**预期结果**: 18个测试（9个算法 × 2种生成）

### 示例 4: 测试所有反演模块

```bash
pytest test/test_watermark_algorithms.py -v -m inversion
```
**预期结果**: 4个测试通过（2个4D测试 + 1个5D测试 + 1个重建测试）

### 示例 5: 在 CI/CD 中运行（跳过耗时测试）

```bash
pytest test/test_watermark_algorithms.py -v \
    -m "not slow" \
    --tb=short \
    --maxfail=3
```

### 示例 6: 生成完整测试报告

```bash
pytest test/test_watermark_algorithms.py -v \
    --html=report.html \
    --cov=watermark \
    --cov=inversions \
    --cov-report=html
```
**输出**:
- `report.html` - 测试报告
- `htmlcov/` - 覆盖率报告

### 示例 7: 调试特定算法的失败

```bash
pytest test/test_watermark_algorithms.py -v \
    --algorithm TR \
    -s \
    --tb=long \
    --pdb
```

### 示例 8: 并行测试以提高速度

```bash
# 安装 pytest-xdist
pip install pytest-xdist

# 并行运行测试
pytest test/test_watermark_algorithms.py -v -n auto
```

## 📊 测试报告

### 查看详细输出

```bash
# 显示详细的测试输出（包括print语句）
pytest test/test_watermark_algorithms.py -v -s

# 显示测试覆盖率
pytest test/test_watermark_algorithms.py -v --cov=watermark --cov=inversions
```

### 生成 HTML 报告

```bash
# 安装 pytest-html
pip install pytest-html

# 生成 HTML 报告
pytest test/test_watermark_algorithms.py -v --html=report.html --self-contained-html
```

## 🔧 故障排除

### 问题 1: 模型加载失败

**错误信息**: `Failed to load image/video model`

**解决方案**:
1. 检查模型路径是否正确
2. 确保有足够的磁盘空间和内存
3. 使用 `--image-model-path` 或 `--video-model-path` 指定本地模型路径

```bash
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /local/path/to/model
```

### 问题 2: CUDA 内存不足

**错误信息**: `CUDA out of memory`

**解决方案**:
1. 减少批处理大小
2. 使用 CPU 运行测试（会自动检测）
3. 一次只测试一个算法：

```bash
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

### 问题 3: 测试超时

**错误信息**: `Test timeout`

**解决方案**:
1. 增加超时时间：在 `pytest.ini` 中修改 `timeout` 值
2. 跳过耗时测试：

```bash
pytest test/test_watermark_algorithms.py -v --skip-generation --skip-detection
```

3. 只运行初始化测试：

```bash
pytest test/test_watermark_algorithms.py -v -k "initialization"
```

### 问题 4: 配置文件未找到

**错误信息**: `Config file not found`

**解决方案**:
1. 确保从项目根目录运行测试
2. 检查 `config/` 目录中是否存在对应的 JSON 配置文件
3. 验证配置文件名称大小写是否正确

### 问题 5: 反演测试失败

**错误信息**: `Failed to invert 4D/5D input`

**解决方案**:
1. 检查设备是否有足够的GPU内存
2. 验证scheduler和unet模型是否正确加载
3. 查看详细错误输出：

```bash
pytest test/test_watermark_algorithms.py -v -s -k inversion
```

## 📈 性能优化

### 测试速度

- **快速测试**（仅初始化）: ~10-30 秒
- **完整测试**（包含生成和检测）: ~10-30 分钟（取决于硬件）
- **反演测试**: ~1-3 分钟（4D）、~5-10 分钟（5D）

### 优化建议

1. **使用 `-k initialization` 进行快速验证**
   ```bash
   pytest test/test_watermark_algorithms.py -v -k initialization
   ```

2. **使用 `--skip-generation` 跳过耗时的生成测试**
   ```bash
   pytest test/test_watermark_algorithms.py -v --skip-generation
   ```

3. **使用 `-n auto` 并行运行测试**
   ```bash
   pip install pytest-xdist
   pytest test/test_watermark_algorithms.py -v -n auto
   ```

4. **使用 `--algorithm` 只测试单个算法**
   ```bash
   pytest test/test_watermark_algorithms.py -v --algorithm TR
   ```

5. **使用 session 级 fixtures 缓存模型**
   - 模型只加载一次，在所有测试间共享
   - 由 `conftest.py` 自动处理

6. **使用 GPU 加速**
   - 测试会自动检测并使用可用的 CUDA 设备
   - 大幅提升测试速度

## 📝 添加新的测试

### 为新的水印算法添加测试

如果你想为新的水印算法添加测试，只需：

1. 在 `watermark/auto_watermark.py` 中注册新算法
2. 在 `config/` 目录中添加配置文件
3. 测试框架会自动发现并测试新算法

**不需要修改任何测试代码！**

### 为反演模块添加新测试

在 `test_watermark_algorithms.py` 中添加新的测试函数：

```python
@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_new_inversion_feature(inversion_type, device, image_pipeline):
    # 测试代码
    pass
```

### 修改测试参数

编辑 `conftest.py` 中的常量：

```python
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_FRAMES = 16
```

或通过命令行参数覆盖默认值。

## ✨ 核心特性

1. ✅ **零冗余设计** - 一个测试文件覆盖所有 11 个算法 + 反演模块
2. ✅ **参数化测试** - 自动为每个算法生成测试用例
3. ✅ **灵活过滤** - 支持按算法、类型、功能过滤
4. ✅ **命令行参数** - 支持自定义模型路径、跳过测试等
5. ✅ **Session 级 Fixtures** - 模型只加载一次，提高效率
6. ✅ **详细文档** - 包含完整的使用说明和示例
7. ✅ **便捷脚本** - 提供友好的命令行工具
8. ✅ **CI/CD 就绪** - 包含 GitHub Actions 配置示例
9. ✅ **可扩展** - 新增算法无需修改测试代码
10. ✅ **错误处理** - 优雅处理未实现的功能
11. ✅ **反演测试** - 完整的4D/5D输入测试和重建验证

## 🎯 测试覆盖总结

### 算法测试矩阵

| 测试类型 | 图像算法 | 视频算法 | 反演模块 | 总计 |
|---------|---------|---------|---------|------|
| 初始化测试 | 9 | 2 | - | 11 |
| 生成测试（带水印） | 9 | 2 | - | 11 |
| 生成测试（不带水印） | 9 | 2 | - | 11 |
| 检测测试 | 9 | 2 | - | 11 |
| 4D反演测试 | - | - | 2 | 2 |
| 5D反演测试 | - | - | 1 | 1 |
| 重建精度测试 | - | - | 1 | 1 |
| **总计** | **36** | **8** | **4** | **48** |

### 反演测试详情

| 测试名称 | 输入维度 | 反演方法 | 测试内容 |
|---------|---------|---------|---------|
| test_inversion_4d_image_input[ddim] | 4D (B,C,H,W) | DDIM | 图像潜在向量反演 |
| test_inversion_4d_image_input[exact] | 4D (B,C,H,W) | Exact | 图像潜在向量反演 |
| test_inversion_5d_video_input[ddim] | 5D (B,F,C,H,W) | DDIM | 视频帧潜在向量反演 |
| test_inversion_reconstruction_accuracy | 4D (B,C,H,W) | DDIM | 前向+反向重建精度 |

**符号说明**:
- B: batch_size
- C: channels (潜在空间通道数，通常为4)
- H: height (潜在空间高度)
- W: width (潜在空间宽度)
- F: num_frames (视频帧数)

## 🤝 贡献指南

### 贡献测试改进

如果你发现测试中的问题或想要改进测试框架，请：

1. 创建 Issue 描述问题或改进建议
2. Fork 项目并创建分支
3. 提交 Pull Request 并附上测试结果
4. 确保所有现有测试仍然通过

### 添加新功能测试

1. 在 `test_watermark_algorithms.py` 中添加新的测试函数
2. 使用 `@pytest.mark.parametrize` 装饰器
3. 使用 `conftest.py` 中的 fixtures
4. 添加适当的测试标记
5. 更新本文档

## 🎓 学习资源

### pytest 相关
- [pytest 官方文档](https://docs.pytest.org/)
- [pytest fixtures 文档](https://docs.pytest.org/en/stable/fixture.html)
- [pytest parametrize 文档](https://docs.pytest.org/en/stable/parametrize.html)
- [pytest 标记文档](https://docs.pytest.org/en/stable/mark.html)

### 项目相关
- MarkDiffusion 项目文档
- `watermark/` 目录下的各个算法实现
- `inversions/` 目录下的反演模块实现
- `config/` 目录下的配置文件

## 💻 CI/CD 集成

### GitHub Actions 示例

参考 `.github_workflows_example.yml` 文件，包含：

1. **快速测试**: 只测试初始化（适合每次提交）
2. **完整测试**: 包含生成和检测（适合PR和发布）
3. **矩阵测试**: 并行测试多个算法

### 本地CI测试

模拟CI环境运行测试：

```bash
# 快速CI测试
pytest test/test_watermark_algorithms.py -v \
    -k initialization \
    --tb=short \
    --maxfail=3

# 完整CI测试
pytest test/test_watermark_algorithms.py -v \
    --html=report.html \
    --cov=watermark \
    --cov=inversions \
    --cov-report=html
```

## 📄 许可证

本测试代码遵循 MarkDiffusion 项目的 Apache 2.0 许可证。

---

**创建日期**: 2025-11-19
**最后更新**: 2025-11-20
**版本**: 2.0.0
**状态**: ✅ 已完成（包含反演测试）
**维护者**: MarkDiffusion Team

🎉 **测试框架已就绪，包含完整的水印算法和反演模块测试！**
