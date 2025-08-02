#!/usr/bin/env python3
"""
Automated dependency update script for Quantum Agent Scheduler.

This script checks for dependency updates, creates pull requests for updates,
and manages dependency security scanning.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import requests


class DependencyUpdater:
    """Manages dependency updates and security scanning."""
    
    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        self.repo_path = Path(repo_path)
        self.dry_run = dry_run
        self.pyproject_path = self.repo_path / "pyproject.toml"
        self.requirements_path = self.repo_path / "requirements.txt"
        
    def _run_command(self, command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages using pip list --outdated."""
        print("Checking for outdated packages...")
        
        code, stdout, stderr = self._run_command([
            "python", "-m", "pip", "list", "--outdated", "--format=json"
        ])
        
        if code != 0:
            print(f"Error checking outdated packages: {stderr}")
            return []
        
        try:
            outdated = json.loads(stdout)
            print(f"Found {len(outdated)} outdated packages")
            return outdated
        except json.JSONDecodeError:
            print("Failed to parse pip list output")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for security vulnerabilities using safety."""
        print("Checking for security vulnerabilities...")
        
        code, stdout, stderr = self._run_command([
            "python", "-m", "safety", "check", "--json"
        ])
        
        vulnerabilities = []
        if stdout:
            try:
                vulnerabilities = json.loads(stdout)
                print(f"Found {len(vulnerabilities)} security vulnerabilities")
            except json.JSONDecodeError:
                print("Failed to parse safety output")
        
        return vulnerabilities
    
    def get_package_info(self, package_name: str) -> Optional[Dict]:
        """Get package information from PyPI."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Failed to get info for {package_name}: {e}")
        return None
    
    def is_safe_update(self, package: Dict[str, str]) -> bool:
        """Determine if a package update is safe (patch or minor version)."""
        current_version = package.get("version", "")
        latest_version = package.get("latest_version", "")
        
        if not current_version or not latest_version:
            return False
        
        try:
            current_parts = [int(x) for x in current_version.split('.')]
            latest_parts = [int(x) for x in latest_version.split('.')]
            
            # Ensure we have at least major.minor.patch
            while len(current_parts) < 3:
                current_parts.append(0)
            while len(latest_parts) < 3:
                latest_parts.append(0)
            
            # Safe if major version is the same and minor version change is small
            if current_parts[0] != latest_parts[0]:
                return False  # Major version change
            
            if latest_parts[1] - current_parts[1] > 1:
                return False  # More than one minor version jump
            
            return True
            
        except (ValueError, IndexError):
            return False
    
    def update_poetry_dependencies(self, packages: List[str], security_only: bool = False) -> bool:
        """Update specific packages using Poetry."""
        if not packages:
            return True
        
        print(f"Updating packages with Poetry: {', '.join(packages)}")
        
        if self.dry_run:
            print("DRY RUN: Would update packages")
            return True
        
        # Update packages
        for package in packages:
            code, stdout, stderr = self._run_command([
                "poetry", "update", package
            ])
            
            if code != 0:
                print(f"Failed to update {package}: {stderr}")
                return False
            else:
                print(f"Successfully updated {package}")
        
        # Run tests after update
        if not security_only:
            print("Running tests after dependency update...")
            code, stdout, stderr = self._run_command([
                "python", "-m", "pytest", "tests/", "-x", "--tb=short"
            ])
            
            if code != 0:
                print("Tests failed after dependency update")
                print(stderr)
                return False
        
        return True
    
    def create_update_branch(self, branch_name: str) -> bool:
        """Create a new branch for dependency updates."""
        if self.dry_run:
            print(f"DRY RUN: Would create branch {branch_name}")
            return True
        
        # Check if branch already exists
        code, stdout, stderr = self._run_command([
            "git", "rev-parse", "--verify", branch_name
        ])
        
        if code == 0:
            print(f"Branch {branch_name} already exists")
            return False
        
        # Create new branch
        code, stdout, stderr = self._run_command([
            "git", "checkout", "-b", branch_name
        ])
        
        if code != 0:
            print(f"Failed to create branch {branch_name}: {stderr}")
            return False
        
        print(f"Created branch {branch_name}")
        return True
    
    def commit_changes(self, message: str) -> bool:
        """Commit changes to git."""
        if self.dry_run:
            print(f"DRY RUN: Would commit with message: {message}")
            return True
        
        # Add changes
        code, stdout, stderr = self._run_command([
            "git", "add", "pyproject.toml", "poetry.lock"
        ])
        
        if code != 0:
            print(f"Failed to add files: {stderr}")
            return False
        
        # Check if there are changes to commit
        code, stdout, stderr = self._run_command([
            "git", "diff", "--cached", "--quiet"
        ])
        
        if code == 0:
            print("No changes to commit")
            return True
        
        # Commit changes
        code, stdout, stderr = self._run_command([
            "git", "commit", "-m", message
        ])
        
        if code != 0:
            print(f"Failed to commit changes: {stderr}")
            return False
        
        print("Committed changes")
        return True
    
    def push_branch(self, branch_name: str) -> bool:
        """Push branch to remote."""
        if self.dry_run:
            print(f"DRY RUN: Would push branch {branch_name}")
            return True
        
        code, stdout, stderr = self._run_command([
            "git", "push", "-u", "origin", branch_name
        ])
        
        if code != 0:
            print(f"Failed to push branch {branch_name}: {stderr}")
            return False
        
        print(f"Pushed branch {branch_name}")
        return True
    
    def create_pull_request(self, branch_name: str, title: str, body: str) -> bool:
        """Create pull request using GitHub CLI."""
        if self.dry_run:
            print(f"DRY RUN: Would create PR: {title}")
            return True
        
        code, stdout, stderr = self._run_command([
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--base", "main",
            "--head", branch_name
        ])
        
        if code != 0:
            print(f"Failed to create pull request: {stderr}")
            return False
        
        print(f"Created pull request: {title}")
        return True
    
    def process_security_updates(self) -> bool:
        """Process security updates with high priority."""
        vulnerabilities = self.check_security_vulnerabilities()
        
        if not vulnerabilities:
            print("No security vulnerabilities found")
            return True
        
        # Group vulnerabilities by package
        vulnerable_packages = {}
        for vuln in vulnerabilities:
            package_name = vuln.get("package_name", "")
            if package_name:
                if package_name not in vulnerable_packages:
                    vulnerable_packages[package_name] = []
                vulnerable_packages[package_name].append(vuln)
        
        if not vulnerable_packages:
            return True
        
        # Create security update branch
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"security-updates-{timestamp}"
        
        if not self.create_update_branch(branch_name):
            return False
        
        # Update vulnerable packages
        packages_to_update = list(vulnerable_packages.keys())
        if not self.update_poetry_dependencies(packages_to_update, security_only=True):
            # Checkout back to main if update failed
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Create commit message
        vuln_count = len(vulnerabilities)
        package_count = len(vulnerable_packages)
        commit_message = f"security: update {package_count} packages to fix {vuln_count} vulnerabilities"
        
        if not self.commit_changes(commit_message):
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Push branch and create PR
        if not self.push_branch(branch_name):
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Create PR body
        pr_body = f"""## Security Updates

This PR updates {package_count} packages to fix {vuln_count} security vulnerabilities.

### Vulnerabilities Fixed:

"""
        
        for package_name, package_vulns in vulnerable_packages.items():
            pr_body += f"\n**{package_name}:**\n"
            for vuln in package_vulns:
                pr_body += f"- {vuln.get('advisory', 'Security vulnerability')} (ID: {vuln.get('id', 'N/A')})\n"
        
        pr_body += """
### Testing
- [ ] All tests pass
- [ ] Security scan shows no remaining vulnerabilities
- [ ] Application starts and functions correctly

**Priority: HIGH** - This PR addresses security vulnerabilities and should be reviewed promptly.
"""
        
        success = self.create_pull_request(
            branch_name,
            f"🔒 Security updates: Fix {vuln_count} vulnerabilities",
            pr_body
        )
        
        # Return to main branch
        self._run_command(["git", "checkout", "main"])
        
        return success
    
    def process_regular_updates(self, max_updates: int = 5) -> bool:
        """Process regular dependency updates."""
        outdated = self.check_outdated_packages()
        
        if not outdated:
            print("All packages are up to date")
            return True
        
        # Filter for safe updates
        safe_updates = [pkg for pkg in outdated if self.is_safe_update(pkg)]
        
        if not safe_updates:
            print("No safe updates available")
            return True
        
        # Limit number of updates per run
        updates_to_process = safe_updates[:max_updates]
        
        print(f"Processing {len(updates_to_process)} regular updates")
        
        # Create update branch
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"dependency-updates-{timestamp}"
        
        if not self.create_update_branch(branch_name):
            return False
        
        # Update packages
        packages_to_update = [pkg["name"] for pkg in updates_to_process]
        if not self.update_poetry_dependencies(packages_to_update):
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Create commit message
        commit_message = f"deps: update {len(packages_to_update)} packages"
        
        if not self.commit_changes(commit_message):
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Push branch and create PR
        if not self.push_branch(branch_name):
            self._run_command(["git", "checkout", "main"])
            return False
        
        # Create PR body
        pr_body = f"""## Dependency Updates

This PR updates {len(packages_to_update)} packages to their latest compatible versions.

### Updated Packages:

"""
        
        for pkg in updates_to_process:
            pr_body += f"- **{pkg['name']}**: {pkg['version']} → {pkg['latest_version']}\n"
        
        pr_body += """
### Testing
- [ ] All tests pass
- [ ] No breaking changes detected
- [ ] Application functions correctly

This PR contains only minor and patch version updates that should be safe to merge.
"""
        
        success = self.create_pull_request(
            branch_name,
            f"⬆️ Update {len(packages_to_update)} dependencies",
            pr_body
        )
        
        # Return to main branch
        self._run_command(["git", "checkout", "main"])
        
        return success
    
    def generate_dependency_report(self) -> str:
        """Generate a dependency status report."""
        outdated = self.check_outdated_packages()
        vulnerabilities = self.check_security_vulnerabilities()
        
        report = f"""# Dependency Status Report

**Generated**: {datetime.now().isoformat()}

## Summary

- **Outdated Packages**: {len(outdated)}
- **Security Vulnerabilities**: {len(vulnerabilities)}

## Outdated Packages

"""
        
        if outdated:
            for pkg in outdated:
                safe = "✅ Safe" if self.is_safe_update(pkg) else "⚠️ Major"
                report += f"- **{pkg['name']}**: {pkg['version']} → {pkg['latest_version']} ({safe})\n"
        else:
            report += "All packages are up to date! 🎉\n"
        
        report += "\n## Security Vulnerabilities\n\n"
        
        if vulnerabilities:
            for vuln in vulnerabilities:
                report += f"- **{vuln.get('package_name', 'Unknown')}**: {vuln.get('advisory', 'Security issue')} (ID: {vuln.get('id', 'N/A')})\n"
        else:
            report += "No security vulnerabilities found! 🔒\n"
        
        report += """
## Recommendations

"""
        
        if vulnerabilities:
            report += "1. 🚨 **Priority**: Address security vulnerabilities immediately\n"
        
        safe_updates = [pkg for pkg in outdated if self.is_safe_update(pkg)]
        if safe_updates:
            report += f"2. ⬆️ **Update**: {len(safe_updates)} safe updates available\n"
        
        major_updates = [pkg for pkg in outdated if not self.is_safe_update(pkg)]
        if major_updates:
            report += f"3. ⚠️ **Review**: {len(major_updates)} major updates require careful review\n"
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated dependency management")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--security-only", action="store_true", help="Only process security updates")
    parser.add_argument("--max-updates", type=int, default=5, help="Maximum number of regular updates per run")
    parser.add_argument("--report", action="store_true", help="Generate dependency report")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(args.repo_path, args.dry_run)
    
    if args.report:
        report = updater.generate_dependency_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
        return
    
    success = True
    
    # Always process security updates first
    print("Processing security updates...")
    if not updater.process_security_updates():
        print("Failed to process security updates")
        success = False
    
    # Process regular updates if not security-only mode
    if not args.security_only:
        print("Processing regular updates...")
        if not updater.process_regular_updates(args.max_updates):
            print("Failed to process regular updates")
            success = False
    
    if success:
        print("Dependency management completed successfully!")
    else:
        print("Some dependency management tasks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()