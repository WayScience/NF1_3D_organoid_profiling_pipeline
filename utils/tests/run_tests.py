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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class TestRunner:
    """Runs featurization tests with various configurations."""

    def __init__(self):
        """Initialize the test runner."""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent
        self.utils_dir = self.project_root / "utils"
        self.test_dir = self.script_dir
        self.console = Console()

    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        self.console.print()
        self.console.print(Panel(text, style="bold blue", expand=False))
        self.console.print()

    def print_status(self, text: str) -> None:
        """Print a status message."""
        self.console.print(f"[green]✓[/green] {text}")

    def print_error(self, text: str) -> None:
        """Print an error message."""
        self.console.print(f"[red]✗[/red] {text}")

    def print_warning(self, text: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {text}")

    def print_info(self, text: str) -> None:
        """Print an info message."""
        self.console.print(f"[cyan]ℹ[/cyan] {text}")

    def check_pytest(self) -> bool:
        """Check if pytest is available."""
        try:
            subprocess.run(
                ["pytest", "--version"], capture_output=True, check=True, timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_package_installed(self) -> bool:
        """Check if the image-analysis-3d package is installed."""
        try:
            subprocess.run(
                [sys.executable, "-c", "import image_analysis_3D"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def check_uv_available(self) -> bool:
        """Check if uv is available."""
        try:
            subprocess.run(
                ["uv", "--version"], capture_output=True, check=True, timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_dependencies(self) -> bool:
        """Install test dependencies."""
        try:
            self.print_warning("pytest not found. Installing dependencies...")
            os.chdir(str(self.utils_dir))

            if self.check_uv_available():
                cmd = ["uv", "pip", "install", "-e", ".[dev]"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
            self.print_status("Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            return False

    def install_package(self) -> bool:
        """Install the image-analysis-3d package in editable mode."""
        try:
            self.print_warning("Package not installed. Installing in editable mode...")
            os.chdir(str(self.utils_dir))

            if self.check_uv_available():
                cmd = ["uv", "pip", "install", "-e", "."]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "-e", "."]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
            self.print_status("Package installed")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install package: {e}")
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

        self.console.print()
        self.print_header(f"Running Tests: {test_type.upper()}")
        self.console.print(f"[dim]Command: {' '.join(args)}[/dim]")
        self.console.print()

        try:
            result = subprocess.run(args, check=False)
            return result.returncode == 0
        except KeyboardInterrupt:
            self.console.print()
            self.print_error("Tests interrupted by user")
            return False

    def print_coverage_summary(self, coverage: bool) -> None:
        """Print coverage report summary."""
        if coverage:
            self.print_header("Coverage Report Generated")
            print(f"HTML Report: {self.test_dir}/htmlcov/index.html")
            self.console.print(f"HTML Report: {self.test_dir}/htmlcov/index.html")
            self.console.print()
            self.console.print("To view the report:")
            if sys.platform == "darwin":
                self.console.print("  open htmlcov/index.html")
            else:
                self.console.print("  xdg-open htmlcov/index.html")

    def print_summary(
        self,
        test_type: str,
        coverage: bool,
        verbose: bool,
        parallel: bool,
    ) -> None:
        """Print test configuration summary."""
        self.print_header("Test Configuration")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Test Type:", test_type)
        table.add_row("Coverage:", str(coverage))
        table.add_row("Verbose:", str(verbose))
        table.add_row("Parallel:", str(parallel))
        table.add_row("", "")
        table.add_row("Project Root:", str(self.project_root))
        table.add_row("Utils Directory:", str(self.utils_dir))
        table.add_row("Test Directory:", str(self.test_dir))

        self.console.print(table)
        self.console.print()

    def run(
        self,
        test_type: str = "all",
        coverage: bool = False,
        verbose: bool = False,
        parallel: bool = False,
    ) -> int:
        """Execute the test runner."""
        # Clear screen and print header
        self.console.clear()

        self.console.print(
            Panel.fit(
                "[bold]Featurization Tests Runner - NF1 Pipeline[/bold]",
                style="blue",
                border_style="blue",
            )
        )

        # Print summary
        self.print_summary(test_type, coverage, verbose, parallel)

        # Check pytest availability
        if not self.check_pytest():
            if not self.install_dependencies():
                return 1

        self.print_status("pytest is available")

        # Check if package is installed
        if not self.check_package_installed():
            if not self.install_package():
                return 1

        self.print_status("Package is installed")
        self.console.print()

        # Run tests
        if self.run_tests(test_type, coverage, verbose, parallel):
            success = True
        else:
            success = False

        # Print coverage summary
        self.print_coverage_summary(coverage)

        # Print final status
        self.console.print()
        if success:
            self.print_header("✓ All Tests Completed Successfully")
            self.console.print("[bold green]No issues found![/bold green]")
        else:
            self.print_header("✗ Tests Failed")
            self.console.print(
                "[bold red]Please review the output above for details.[/bold red]"
            )

        self.console.print()
        return 0 if success else 1


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
sys.exit(
    runner.run(
        test_type=args.test_type or "all",
        coverage=args.coverage,
        verbose=args.verbose,
        parallel=args.parallel,
    )
)
