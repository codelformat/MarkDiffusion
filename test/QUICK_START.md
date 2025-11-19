# 🚀 快速开始指南

## 一分钟上手

### 1. 安装测试依赖
```bash
pip install -r test/requirements-test.txt
```

### 2. 运行测试

#### 使用 pytest 直接运行
```bash
# 测试所有算法
pytest test/test_watermark_algorithms.py -v

# 测试特定算法
pytest test/test_watermark_algorithms.py -v --algorithm TR
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

## 📋 常用命令速查表

| 需求 | 命令 |
|------|------|
| 测试所有算法 | `pytest test/test_watermark_algorithms.py -v` |
| 测试图像算法 | `pytest test/test_watermark_algorithms.py -v -m image` |
| 测试视频算法 | `pytest test/test_watermark_algorithms.py -v -m video` |
| 测试 TR 算法 | `pytest test/test_watermark_algorithms.py -v -k TR` |
| 快速测试 | `pytest test/test_watermark_algorithms.py -v -k initialization` |
| 跳过生成测试 | `pytest test/test_watermark_algorithms.py -v --skip-generation` |
| 并行运行 | `pytest test/test_watermark_algorithms.py -v -n auto` |
| 生成报告 | `pytest test/test_watermark_algorithms.py -v --html=report.html` |

## 🎯 测试范围

### 图像水印算法 (9个)
TR, GS, PRC, RI, SEAL, ROBIN, WIND, GM, SFW

### 视频水印算法 (2个)
VideoShield, VideoMark

## 🧪 测试内容

每个算法会进行以下测试：

1. ✅ **初始化测试** - 验证算法能否正确加载
2. ✅ **生成测试** - 验证能否生成带水印/不带水印的媒体
3. ✅ **检测测试** - 验证能否正确检测水印

## ⚡ 性能提示

- **快速验证**: 使用 `-k initialization` 只测试初始化（秒级完成）
- **跳过耗时**: 使用 `--skip-generation` 或 `--skip-detection`
- **并行执行**: 使用 `-n auto` 并行运行测试
- **单个算法**: 使用 `--algorithm NAME` 只测试一个算法

## 🔧 自定义配置

### 指定模型路径
```bash
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /path/to/your/model
```

### 组合使用
```bash
# 快速测试所有图像算法的初始化
pytest test/test_watermark_algorithms.py -v -m image -k initialization

# 测试 TR 算法但跳过生成
pytest test/test_watermark_algorithms.py -v --algorithm TR --skip-generation
```

## 📊 查看结果

测试结果会显示：
- ✅ 通过的测试
- ❌ 失败的测试
- ⚠️ 跳过的测试（如未实现的功能）

## 💡 故障排除

### 模型加载失败
```bash
# 使用本地模型路径
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /local/path/to/model
```

### 内存不足
```bash
# 一次只测试一个算法
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

### 测试超时
```bash
# 只运行快速测试
pytest test/test_watermark_algorithms.py -v -k initialization
```

## 📚 更多信息

详细文档请查看 [README.md](README.md)
