#!/usr/bin/env python3
"""
Automated metrics collection script for Quantum Agent Scheduler.

This script collects various metrics about the project health, performance,
and compliance, updating the project-metrics.json file automatically.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import argparse


class MetricsCollector:
    """Collects and updates project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or create new structure."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        else:
            # Return default structure if file doesn't exist
            return self._create_default_metrics()
    
    def _create_default_metrics(self) -> Dict[str, Any]:
        """Create default metrics structure."""
        return {
            "project": {
                "name": "quantum-agent-scheduler",
                "last_updated": datetime.utcnow().isoformat() + "Z"
            },
            "repository_health": {
                "last_collected": datetime.utcnow().isoformat() + "Z",
                "health_score": 0.0
            }
        }
    
    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            return ""
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("Collecting Git metrics...")
        
        # Commit frequency
        commits_last_week = self._run_command([
            "git", "rev-list", "--count", "--since=1.week", "HEAD"
        ])
        
        # Contributors
        contributors = self._run_command([
            "git", "shortlog", "-sn", "--since=30.days"
        ])
        active_contributors = len(contributors.split('\n')) if contributors else 0
        
        # Recent commits for trend analysis
        commit_dates = self._run_command([
            "git", "log", "--format=%ci", "--since=12.weeks"
        ]).split('\n')
        
        weekly_commits = self._calculate_weekly_commits(commit_dates)
        
        return {
            "commit_frequency": {
                "commits_per_week": int(commits_last_week) if commits_last_week else 0,
                "last_12_weeks": weekly_commits
            },
            "contributor_activity": {
                "active_contributors": active_contributors,
                "last_12_weeks": [active_contributors] * 12  # Simplified
            }
        }
    
    def _calculate_weekly_commits(self, commit_dates: List[str]) -> List[int]:
        """Calculate commits per week for the last 12 weeks."""
        if not commit_dates or commit_dates == ['']:
            return [0] * 12
            
        now = datetime.now()
        weekly_counts = [0] * 12
        
        for date_str in commit_dates:
            if not date_str:
                continue
            try:
                commit_date = datetime.fromisoformat(date_str.split()[0])
                weeks_ago = (now - commit_date).days // 7
                if 0 <= weeks_ago < 12:
                    weekly_counts[11 - weeks_ago] += 1
            except (ValueError, IndexError):
                continue
                
        return weekly_counts
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect testing metrics."""
        print("Collecting test metrics...")
        
        # Run pytest with coverage
        coverage_output = self._run_command([
            "python", "-m", "pytest", "--cov=src", "--cov-report=json", "--quiet"
        ])
        
        coverage_data = {}
        coverage_file = self.repo_path / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
        
        coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
        
        # Test success rate (simplified)
        test_result = self._run_command([
            "python", "-m", "pytest", "--tb=no", "-q"
        ])
        
        # Parse test results
        if "FAILED" in test_result:
            test_success_rate = 0
        elif "PASSED" in test_result or "passed" in test_result:
            test_success_rate = 100
        else:
            test_success_rate = 0
            
        return {
            "test_coverage": coverage_percent,
            "test_success_rate": test_success_rate
        }
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("Collecting code quality metrics...")
        
        # Run ruff for linting
        ruff_output = self._run_command([
            "python", "-m", "ruff", "check", "src/", "--format=json"
        ])
        
        lint_issues = 0
        if ruff_output:
            try:
                ruff_data = json.loads(ruff_output)
                lint_issues = len(ruff_data)
            except json.JSONDecodeError:
                pass
        
        # Count lines of code
        loc_output = self._run_command([
            "find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
        ])
        
        total_loc = 0
        if loc_output:
            lines = loc_output.split('\n')
            for line in lines:
                if 'total' in line.lower():
                    try:
                        total_loc = int(line.split()[0])
                        break
                    except (ValueError, IndexError):
                        pass
        
        # Calculate technical debt ratio (simplified)
        technical_debt_ratio = (lint_issues / max(total_loc, 1)) * 100
        
        return {
            "code_duplication": 0,  # Would need additional tooling
            "technical_debt_ratio": technical_debt_ratio,
            "maintainability_index": max(0, 100 - technical_debt_ratio),
            "lint_issues": lint_issues,
            "lines_of_code": total_loc
        }
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        print("Collecting security metrics...")
        
        # Run safety check for dependency vulnerabilities
        safety_output = self._run_command([
            "python", "-m", "safety", "check", "--json"
        ])
        
        vulnerabilities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                for vuln in safety_data:
                    severity = vuln.get('severity', 'unknown').lower()
                    if severity in vulnerabilities:
                        vulnerabilities[severity] += 1
            except json.JSONDecodeError:
                pass
        
        # Check for secrets (simplified)
        secrets_check = self._run_command([
            "grep", "-r", "-i", "password\\|secret\\|key\\|token", "src/", "--include=*.py"
        ])
        
        # Filter out obvious false positives
        secret_lines = [line for line in secrets_check.split('\n') 
                       if line and 'test' not in line.lower() and 'example' not in line.lower()]
        
        return {
            "dependency_vulnerabilities": vulnerabilities,
            "secrets_exposure": {
                "exposed_secrets": len(secret_lines),
                "last_scan": datetime.utcnow().isoformat() + "Z"
            }
        }
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        print("Collecting documentation metrics...")
        
        # Count documentation files
        doc_files = list(self.repo_path.glob("**/*.md"))
        doc_count = len(doc_files)
        
        # Count Python files
        py_files = list(self.repo_path.glob("src/**/*.py"))
        py_count = len(py_files)
        
        # Simple documentation coverage (files with docstrings)
        documented_files = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        documented_files += 1
            except Exception:
                pass
        
        doc_coverage = (documented_files / max(py_count, 1)) * 100
        
        # Check README quality (basic metrics)
        readme_file = self.repo_path / "README.md"
        readme_quality = 0
        if readme_file.exists():
            with open(readme_file, 'r') as f:
                readme_content = f.read()
                # Basic quality indicators
                quality_indicators = [
                    "installation" in readme_content.lower(),
                    "usage" in readme_content.lower(),
                    "example" in readme_content.lower(),
                    "license" in readme_content.lower(),
                    "contributing" in readme_content.lower(),
                    len(readme_content) > 1000,  # Substantial content
                    readme_content.count('```') >= 2,  # Code examples
                    readme_content.count('#') >= 3,  # Multiple sections
                ]
                readme_quality = (sum(quality_indicators) / len(quality_indicators)) * 100
        
        return {
            "documentation_coverage": doc_coverage,
            "api_documentation": 90,  # Would need API doc analysis
            "readme_quality": readme_quality,
            "total_doc_files": doc_count,
            "documented_py_files": documented_files
        }
    
    def collect_build_metrics(self) -> Dict[str, Any]:
        """Collect build and CI metrics."""
        print("Collecting build metrics...")
        
        # Try to run build command
        build_success = True
        try:
            self._run_command(["python", "-m", "build"])
        except:
            build_success = False
        
        return {
            "build_success_rate": 100 if build_success else 0,
            "deployment_frequency": 0,  # Would need CI/CD integration
            "mean_time_to_recovery": 0  # Would need incident data
        }
    
    def calculate_health_score(self) -> float:
        """Calculate overall repository health score."""
        categories = self.metrics.get("repository_health", {}).get("categories", {})
        
        total_weighted_score = 0
        total_weight = 0
        
        for category_name, category_data in categories.items():
            weight = category_data.get("weight", 0)
            metrics = category_data.get("metrics", {})
            
            category_score = 0
            metric_count = 0
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "current" in metric_data and "target" in metric_data:
                    current = metric_data["current"]
                    target = metric_data["target"]
                    
                    # Calculate metric score (0-100)
                    if target > 0:
                        if metric_name in ["security_vulnerabilities", "dependency_vulnerabilities", 
                                         "secrets_exposure", "technical_debt_ratio"]:
                            # Lower is better for these metrics
                            metric_score = max(0, 100 - (current / target * 100))
                        else:
                            # Higher is better for most metrics
                            metric_score = min(100, (current / target) * 100)
                    else:
                        metric_score = 100 if current == target else 0
                    
                    category_score += metric_score
                    metric_count += 1
            
            if metric_count > 0:
                category_score = category_score / metric_count
                total_weighted_score += category_score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0
    
    def update_metrics(self):
        """Update all metrics."""
        print("Starting metrics collection...")
        
        # Update timestamp
        self.metrics["repository_health"]["last_collected"] = datetime.utcnow().isoformat() + "Z"
        
        # Collect various metrics
        try:
            git_metrics = self.collect_git_metrics()
            test_metrics = self.collect_test_metrics()
            quality_metrics = self.collect_code_quality_metrics()
            security_metrics = self.collect_security_metrics()
            doc_metrics = self.collect_documentation_metrics()
            build_metrics = self.collect_build_metrics()
            
            # Update metrics structure
            categories = self.metrics.setdefault("repository_health", {}).setdefault("categories", {})
            
            # Update code quality metrics
            if "code_quality" in categories:
                code_quality = categories["code_quality"]["metrics"]
                if "test_coverage" in code_quality:
                    code_quality["test_coverage"]["current"] = test_metrics["test_coverage"]
                if "technical_debt_ratio" in code_quality:
                    code_quality["technical_debt_ratio"]["current"] = quality_metrics["technical_debt_ratio"]
                if "maintainability_index" in code_quality:
                    code_quality["maintainability_index"]["current"] = quality_metrics["maintainability_index"]
            
            # Update security metrics
            if "security" in categories:
                security = categories["security"]["metrics"]
                if "dependency_vulnerabilities" in security:
                    security["dependency_vulnerabilities"].update(security_metrics["dependency_vulnerabilities"])
                if "secrets_exposure" in security:
                    security["secrets_exposure"]["exposed_secrets"] = security_metrics["secrets_exposure"]["exposed_secrets"]
            
            # Update documentation metrics
            if "documentation" in categories:
                doc = categories["documentation"]["metrics"]
                if "documentation_coverage" in doc:
                    doc["documentation_coverage"]["current"] = doc_metrics["documentation_coverage"]
                if "readme_quality" in doc:
                    doc["readme_quality"]["current"] = doc_metrics["readme_quality"]
            
            # Update activity metrics
            if "activity" in categories:
                activity = categories["activity"]["metrics"]
                if "commit_frequency" in activity:
                    activity["commit_frequency"]["commits_per_week"] = git_metrics["commit_frequency"]["commits_per_week"]
                    activity["commit_frequency"]["last_12_weeks"] = git_metrics["commit_frequency"]["last_12_weeks"]
                if "contributor_activity" in activity:
                    activity["contributor_activity"]["active_contributors"] = git_metrics["contributor_activity"]["active_contributors"]
            
            # Update performance metrics
            if "performance" in categories:
                performance = categories["performance"]["metrics"]
                if "build_success_rate" in performance:
                    performance["build_success_rate"]["current"] = build_metrics["build_success_rate"]
                if "test_success_rate" in performance:
                    performance["test_success_rate"]["current"] = test_metrics["test_success_rate"]
            
            # Calculate overall health score
            health_score = self.calculate_health_score()
            self.metrics["repository_health"]["health_score"] = round(health_score, 1)
            
            print(f"Repository health score: {health_score:.1f}/100")
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def save_metrics(self):
        """Save metrics to file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {self.metrics_file}")
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        health_score = self.metrics.get("repository_health", {}).get("health_score", 0)
        last_collected = self.metrics.get("repository_health", {}).get("last_collected", "Unknown")
        
        report = f"""
# Repository Health Report

**Overall Health Score**: {health_score}/100
**Last Updated**: {last_collected}

## Key Metrics

### Code Quality
- Test Coverage: {self._get_metric_value('code_quality', 'test_coverage', 'current', 0)}%
- Technical Debt Ratio: {self._get_metric_value('code_quality', 'technical_debt_ratio', 'current', 0)}%
- Maintainability Index: {self._get_metric_value('code_quality', 'maintainability_index', 'current', 0)}/100

### Security
- Dependency Vulnerabilities: {self._get_security_vuln_count()}
- Exposed Secrets: {self._get_metric_value('security', 'secrets_exposure', 'exposed_secrets', 0)}

### Documentation
- Documentation Coverage: {self._get_metric_value('documentation', 'documentation_coverage', 'current', 0)}%
- README Quality: {self._get_metric_value('documentation', 'readme_quality', 'current', 0)}%

### Activity
- Commits per Week: {self._get_metric_value('activity', 'commit_frequency', 'commits_per_week', 0)}
- Active Contributors: {self._get_metric_value('activity', 'contributor_activity', 'active_contributors', 0)}

### Performance
- Build Success Rate: {self._get_metric_value('performance', 'build_success_rate', 'current', 0)}%
- Test Success Rate: {self._get_metric_value('performance', 'test_success_rate', 'current', 0)}%

## Recommendations

"""
        
        # Add recommendations based on metrics
        if health_score < 70:
            report += "- 🚨 Overall health score is below 70. Focus on improving key metrics.\n"
        
        test_coverage = self._get_metric_value('code_quality', 'test_coverage', 'current', 0)
        if test_coverage < 80:
            report += f"- 📝 Test coverage is {test_coverage}%. Aim for 80%+.\n"
        
        vuln_count = self._get_security_vuln_count()
        if vuln_count > 0:
            report += f"- 🔒 {vuln_count} security vulnerabilities found. Address high/critical issues first.\n"
        
        if self._get_metric_value('activity', 'commit_frequency', 'commits_per_week', 0) < 5:
            report += "- 📈 Low commit frequency. Consider more regular development activity.\n"
        
        return report
    
    def _get_metric_value(self, category: str, metric: str, field: str, default: Any) -> Any:
        """Helper to safely get metric values."""
        return (self.metrics.get("repository_health", {})
                .get("categories", {})
                .get(category, {})
                .get("metrics", {})
                .get(metric, {})
                .get(field, default))
    
    def _get_security_vuln_count(self) -> int:
        """Get total security vulnerability count."""
        vuln_data = (self.metrics.get("repository_health", {})
                    .get("categories", {})
                    .get("security", {})
                    .get("metrics", {})
                    .get("dependency_vulnerabilities", {}))
        
        return sum([
            vuln_data.get("critical", 0),
            vuln_data.get("high", 0),
            vuln_data.get("medium", 0),
            vuln_data.get("low", 0)
        ])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_path)
    
    # Update metrics
    collector.update_metrics()
    collector.save_metrics()
    
    # Generate report if requested
    if args.report:
        report = collector.generate_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    print("Metrics collection completed successfully!")


if __name__ == "__main__":
    main()