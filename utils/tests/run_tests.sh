#!/bin/bash

# Featurization Tests Runner Script
# Comprehensive test execution with various options

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UTILS_DIR="$PROJECT_ROOT/utils"

# Default values
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_COVERAGE=false
RUN_PERFORMANCE=false
VERBOSE=false
PARALLEL=false

# Function to print usage
usage() {
    cat << EOF
${BLUE}Featurization Tests Runner${NC}

Usage: $0 [OPTIONS]

Options:
    -u, --unit              Run only unit tests
    -i, --integration       Run only integration tests
    -c, --coverage          Generate coverage report
    -p, --performance       Run performance tests only
    -v, --verbose           Verbose output
    -j, --parallel          Run tests in parallel (faster)
    -a, --all               Run all tests (default)
    -h, --help              Show this help message

Examples:
    # Run all tests with coverage
    $0 --coverage

    # Run only unit tests
    $0 --unit

    # Run all tests in parallel
    $0 --parallel

    # Run with verbose output
    $0 --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=false
            shift
            ;;
        -i|--integration)
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=true
            shift
            ;;
        -c|--coverage)
            RUN_COVERAGE=true
            shift
            ;;
        -p|--performance)
            RUN_PERFORMANCE=true
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--parallel)
            PARALLEL=true
            shift
            ;;
        -a|--all)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to create and activate a uv test environment
setup_uv_env() {
    if ! command -v uv &> /dev/null; then
        print_error "uv not found. Please install uv first."
        exit 1
    fi

    local venv_dir="$PROJECT_ROOT/.venv-test"

    if [ ! -d "$venv_dir" ]; then
        print_status "Creating uv test environment at $venv_dir"
        uv venv "$venv_dir"
    fi

    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"

    print_status "Installing project dependencies"
    (cd "$PROJECT_ROOT" && uv pip install -e .) > /dev/null 2>&1 || {
        print_error "Failed to install project dependencies"
        exit 1
    }

    print_status "Installing utils test dependencies"
    uv pip install -e "${UTILS_DIR}[dev]" > /dev/null 2>&1 || {
        print_error "Failed to install dependencies"
        exit 1
    }
}

# Function to run tests
run_tests() {
    cd "$SCRIPT_DIR"

    print_header "Running Featurization Tests"

    local pytest_args=("-v")
    local test_files=()

    # Add verbose flag
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi

    # Add parallel flag
    if [ "$PARALLEL" = true ]; then
        # Check if pytest-xdist is available
        if python -c "import xdist" 2>/dev/null; then
            pytest_args+=("-n" "auto")
            print_status "Running tests in parallel"
        else
            echo -e "${YELLOW}⚠${NC} pytest-xdist not installed, running sequentially"
            echo "  Install with: pip install pytest-xdist"
        fi
    fi

    # Determine which tests to run
    if [ "$RUN_PERFORMANCE" = true ]; then
        print_header "Performance Tests"
        test_files=("test_featurization_integration.py::TestPerformance")
        echo -e "${YELLOW}Running performance benchmarks...${NC}"
    elif [ "$RUN_UNIT_TESTS" = true ] && [ "$RUN_INTEGRATION_TESTS" = false ]; then
        print_header "Unit Tests"
        test_files=("test_featurization_utils.py")
        echo -e "${YELLOW}Running unit tests...${NC}"
    elif [ "$RUN_INTEGRATION_TESTS" = true ] && [ "$RUN_UNIT_TESTS" = false ]; then
        print_header "Integration Tests"
        test_files=("test_featurization_integration.py")
        echo -e "${YELLOW}Running integration tests...${NC}"
    else
        print_header "All Tests"
        test_files=("test_featurization_utils.py" "test_featurization_integration.py")
        echo -e "${YELLOW}Running all tests...${NC}"
    fi

    # Add coverage reporting
    if [ "$RUN_COVERAGE" = true ]; then
        pytest_args+=("--cov=src.image_analysis_3D.featurization_utils" "--cov-report=html" "--cov-report=term-missing")
        echo -e "${YELLOW}Coverage report will be generated...${NC}"
    fi

    # Run pytest
    if python -m pytest "${pytest_args[@]}" "${test_files[@]}"; then
        print_status "All tests passed!"
        return 0
    else
        print_error "Tests failed!"
        return 1
    fi
}

# Function to generate coverage report summary
print_coverage_summary() {
    if [ "$RUN_COVERAGE" = true ]; then
        print_header "Coverage Report"
        echo -e "${GREEN}Coverage report generated:${NC}"
        echo "  HTML Report: htmlcov/index.html"
        echo ""
        echo "To view the report:"
        if [ "$(uname)" = "Darwin" ]; then
            echo "  open htmlcov/index.html"
        else
            echo "  xdg-open htmlcov/index.html"
        fi
    fi
}

# Function to print summary
print_summary() {
    print_header "Test Summary"

    echo "Configuration:"
    echo "  Unit Tests:          $RUN_UNIT_TESTS"
    echo "  Integration Tests:   $RUN_INTEGRATION_TESTS"
    echo "  Coverage Report:     $RUN_COVERAGE"
    echo "  Performance Tests:   $RUN_PERFORMANCE"
    echo "  Verbose:             $VERBOSE"
    echo "  Parallel:            $PARALLEL"
    echo ""
    echo "Project:"
    echo "  Utils Directory:     $UTILS_DIR"
    echo "  Test Directory:      $SCRIPT_DIR"
    echo ""
}

# Main execution
main() {
    clear

    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       Featurization Tests Runner - NF1 Pipeline          ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    print_summary

    # Create and activate uv test environment
    setup_uv_env
    print_status "uv test environment ready"
    echo ""

    # Run tests
    if run_tests; then
        test_exit_code=0
    else
        test_exit_code=1
    fi

    # Print coverage summary
    print_coverage_summary

    # Print final status
    echo ""
    if [ $test_exit_code -eq 0 ]; then
        print_header "✓ All Tests Completed Successfully"
        echo -e "${GREEN}No issues found!${NC}"
    else
        print_header "✗ Tests Failed"
        echo -e "${RED}Please review the output above for details.${NC}"
    fi

    echo ""

    exit $test_exit_code
}

# Run main function
main
