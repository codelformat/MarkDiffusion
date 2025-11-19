#!/bin/bash
# Convenience script for running MarkDiffusion watermark algorithm tests
# Usage: ./test/run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
ALGORITHM=""
MODEL_PATH=""
EXTRA_ARGS=""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --type TYPE         Test type: all, image, video, quick (default: all)
    -a, --algorithm NAME    Test specific algorithm (e.g., TR, VideoShield)
    -m, --model-path PATH   Custom model path
    --skip-generation       Skip generation tests
    --skip-detection        Skip detection tests
    --parallel              Run tests in parallel
    --coverage              Generate coverage report
    --html                  Generate HTML report

Examples:
    # Run all tests
    $0

    # Run only image watermark tests
    $0 --type image

    # Test specific algorithm
    $0 --algorithm TR

    # Quick test (initialization only)
    $0 --type quick

    # Run with coverage report
    $0 --coverage

    # Run tests in parallel
    $0 --parallel

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --skip-generation)
            EXTRA_ARGS="$EXTRA_ARGS --skip-generation"
            shift
            ;;
        --skip-detection)
            EXTRA_ARGS="$EXTRA_ARGS --skip-detection"
            shift
            ;;
        --parallel)
            EXTRA_ARGS="$EXTRA_ARGS -n auto"
            shift
            ;;
        --coverage)
            EXTRA_ARGS="$EXTRA_ARGS --cov=watermark --cov-report=html --cov-report=term"
            shift
            ;;
        --html)
            EXTRA_ARGS="$EXTRA_ARGS --html=test_report.html --self-contained-html"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Please install it with:"
    echo "  pip install -r test/requirements-test.txt"
    exit 1
fi

# Build pytest command
PYTEST_CMD="pytest test/test_watermark_algorithms.py -v"

# Add test type filter
case $TEST_TYPE in
    all)
        print_info "Running all tests"
        ;;
    image)
        print_info "Running image watermark tests"
        PYTEST_CMD="$PYTEST_CMD -m image"
        ;;
    video)
        print_info "Running video watermark tests"
        PYTEST_CMD="$PYTEST_CMD -m video"
        ;;
    quick)
        print_info "Running quick tests (initialization only)"
        PYTEST_CMD="$PYTEST_CMD -k initialization"
        ;;
    *)
        print_error "Invalid test type: $TEST_TYPE"
        show_usage
        exit 1
        ;;
esac

# Add algorithm filter
if [ -n "$ALGORITHM" ]; then
    print_info "Testing algorithm: $ALGORITHM"
    PYTEST_CMD="$PYTEST_CMD --algorithm $ALGORITHM"
fi

# Add model path
if [ -n "$MODEL_PATH" ]; then
    print_info "Using model path: $MODEL_PATH"
    PYTEST_CMD="$PYTEST_CMD --image-model-path $MODEL_PATH"
fi

# Add extra arguments
PYTEST_CMD="$PYTEST_CMD $EXTRA_ARGS"

# Print command
print_info "Executing: $PYTEST_CMD"
echo ""

# Run tests
if eval $PYTEST_CMD; then
    echo ""
    print_info "Tests completed successfully!"
    exit 0
else
    echo ""
    print_error "Tests failed!"
    exit 1
fi
