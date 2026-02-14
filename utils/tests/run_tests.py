#!/usr/bin/env python
"""
Featurization Tests Runner Script (Python version)

Comprehensive test execution with various options for the NF1 3D organoid profiling pipeline.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class TestRunner:
    """Runs featurization tests with various configurations."""

    def __init__(self):
        """Initialize the test runner."""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent
        self.utils_dir = self.project_root / "utils"
        self.test_dir = self.script_dir

    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        print()
        print(f"{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.BLUE}{text:^60}{Colors.ENDC}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
        print()

    def print_status(self, text: str) -> None:
        """Print a status message."""
        print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")

    def print_error(self, text: str) -> None:
        """Print an error message."""
        print(f"{Colors.RED}✗{Colors.ENDC} {text}")

    def print_warning(self, text: str) -> None:
        """Print a warning message."""
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")

    def print_info(self, text: str) -> None:
        """Print an info message."""
        print(f"{Colors.CYAN}ℹ{Colors.ENDC} {text}")

    def check_pytest(self) -> bool:
        """Check if pytest is available."""
        try:
            subprocess.run(
                ["pytest", "--version"], capture_output=True, check=True, timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_dependencies(self) -> bool:
        """Install test dependencies."""
        try:
            self.print_warning("pytest not found. Installing dependencies...")
            os.chdir(str(self.utils_dir))
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
                check=True,
                capture_output=True,
                timeout=120,
            )
            self.print_status("Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            return False

    def check_xdist(self) -> bool:
        """Check if pytest-xdist is available."""
        try:
            subprocess.run(
                [sys.executable, "-c", "import xdist"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def build_pytest_args(
        self,
        test_type: str,
        coverage: bool = False,
        verbose: bool = False,
        parallel: bool = False,
    ) -> tuple[List[str], str]:
        """Build pytest arguments based on options."""
        args = ["pytest", "-v"]
        test_file = ""

        if verbose:
            args.append("-vv")

        if parallel:
            if self.check_xdist():
                args.extend(["-n", "auto"])
                self.print_status("Running tests in parallel")
            else:
                self.print_warning("pytest-xdist not installed, running sequentially")
                self.print_info("Install with: pip install pytest-xdist")

        # Determine test file
        if test_type == "unit":
            test_file = "test_featurization_utils.py"
            self.print_info("Running unit tests")
        elif test_type == "integration":
            test_file = "test_featurization_integration.py"
            self.print_info("Running integration tests")
        elif test_type == "performance":
            test_file = "test_featurization_integration.py::TestPerformance"
            self.print_info("Running performance benchmarks")
        else:  # all
            test_file = "test_featurization_utils.py test_featurization_integration.py"
            self.print_info("Running all tests")

        # Add coverage arguments
        if coverage:
            args.extend(
                [
                    "--cov=src.image_analysis_3D.featurization_utils",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                ]
            )
            self.print_info("Coverage report will be generated")

        return args, test_file

    def run_tests(
        self,
        test_type: str = "all",
        coverage: bool = False,
        verbose: bool = False,
        parallel: bool = False,
    ) -> bool:
        """Run the tests."""
        os.chdir(str(self.test_dir))

        # Build pytest arguments
        args, test_file = self.build_pytest_args(
            test_type, coverage=coverage, verbose=verbose, parallel=parallel
        )

        # Add test file(s)
        args.extend(test_file.split())

        print()
        self.print_header(f"Running Tests: {test_type.upper()}")
        print(f"Command: {' '.join(args)}")
        print()

        try:
            result = subprocess.run(args, check=False)
            return result.returncode == 0
        except KeyboardInterrupt:
            print()
            self.print_error("Tests interrupted by user")
            return False

    def print_coverage_summary(self, coverage: bool) -> None:
        """Print coverage report summary."""
        if coverage:
            self.print_header("Coverage Report Generated")
            print(f"HTML Report: {self.test_dir}/htmlcov/index.html")
            print()
            print("To view the report:")
            if sys.platform == "darwin":
                print("  open htmlcov/index.html")
            else:
                print("  xdg-open htmlcov/index.html")
            print()

    def print_summary(
        self,
        test_type: str,
        coverage: bool,
        verbose: bool,
        parallel: bool,
    ) -> None:
        """Print test configuration summary."""
        self.print_header("Test Configuration")

        print(f"Test Type:       {test_type}")
        print(f"Coverage:        {coverage}")
        print(f"Verbose:         {verbose}")
        print(f"Parallel:        {parallel}")
        print()
        print(f"Project Root:    {self.project_root}")
        print(f"Utils Directory: {self.utils_dir}")
        print(f"Test Directory:  {self.test_dir}")
        print()

    def run(
        self,
        test_type: str = "all",
        coverage: bool = False,
        verbose: bool = False,
        parallel: bool = False,
    ) -> int:
        """Execute the test runner."""
        # Clear screen and print header
        os.system("clear" if sys.platform != "win32" else "cls")

        print(f"{Colors.BLUE}╔{'═' * 58}╗{Colors.ENDC}")
        print(
            f"{Colors.BLUE}║{'Featurization Tests Runner - NF1 Pipeline':^58}║{Colors.ENDC}"
        )
        print(f"{Colors.BLUE}╚{'═' * 58}╝{Colors.ENDC}")

        # Print summary
        self.print_summary(test_type, coverage, verbose, parallel)

        # Check pytest availability
        if not self.check_pytest():
            if not self.install_dependencies():
                return 1

        self.print_status("pytest is available")
        print()

        # Run tests
        if self.run_tests(test_type, coverage, verbose, parallel):
            success = True
        else:
            success = False

        # Print coverage summary
        self.print_coverage_summary(coverage)

        # Print final status
        print()
        if success:
            self.print_header("✓ All Tests Completed Successfully")
            print(f"{Colors.GREEN}No issues found!{Colors.ENDC}")
        else:
            self.print_header("✗ Tests Failed")
            print(
                f"{Colors.RED}Please review the output above for details.{Colors.ENDC}"
            )

        print()
        return 0 if success else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run featurization tests for NF1 3D organoid profiling pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  %(prog)s

  # Run only unit tests
  %(prog)s --unit

  # Run with coverage report
  %(prog)s --coverage

  # Run in parallel (faster)
  %(prog)s --parallel

  # Run verbose output
  %(prog)s --verbose

  # Run performance tests
  %(prog)s --performance
        """,
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "-u",
        "--unit",
        action="store_const",
        const="unit",
        dest="test_type",
        help="Run only unit tests",
    )
    test_group.add_argument(
        "-i",
        "--integration",
        action="store_const",
        const="integration",
        dest="test_type",
        help="Run only integration tests",
    )
    test_group.add_argument(
        "-p",
        "--performance",
        action="store_const",
        const="performance",
        dest="test_type",
        help="Run only performance tests",
    )

    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-j", "--parallel", action="store_true", help="Run tests in parallel (faster)"
    )

    args = parser.parse_args()

    runner = TestRunner()
    return runner.run(
        test_type=args.test_type or "all",
        coverage=args.coverage,
        verbose=args.verbose,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    sys.exit(main())
