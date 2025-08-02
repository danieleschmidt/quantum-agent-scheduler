#!/usr/bin/env python3
"""
Repository maintenance automation script.

This script performs various maintenance tasks like cleaning up branches,
updating documentation, and running health checks.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import re


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        self.repo_path = Path(repo_path)
        self.dry_run = dry_run
        
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
    
    def cleanup_merged_branches(self, keep_recent_days: int = 7) -> List[str]:
        """Clean up merged branches older than specified days."""
        print("Cleaning up merged branches...")
        
        # Get list of merged branches
        code, stdout, stderr = self._run_command([
            "git", "branch", "--merged", "main"
        ])
        
        if code != 0:
            print(f"Failed to get merged branches: {stderr}")
            return []
        
        merged_branches = []
        for line in stdout.strip().split('\n'):
            branch = line.strip().lstrip('*').strip()
            if branch and branch not in ['main', 'master', 'develop']:
                merged_branches.append(branch)
        
        # Get branch ages
        cutoff_date = datetime.now() - timedelta(days=keep_recent_days)
        branches_to_delete = []
        
        for branch in merged_branches:
            # Get last commit date for branch
            code, stdout, stderr = self._run_command([
                "git", "log", "-1", "--format=%ci", branch
            ])
            
            if code == 0 and stdout:
                try:
                    last_commit = datetime.fromisoformat(stdout.strip().split('+')[0])
                    if last_commit < cutoff_date:
                        branches_to_delete.append(branch)
                except ValueError:
                    # If we can't parse the date, skip this branch
                    continue
        
        # Delete old merged branches
        deleted_branches = []
        for branch in branches_to_delete:
            if self.dry_run:
                print(f"DRY RUN: Would delete branch {branch}")
                deleted_branches.append(branch)
            else:
                code, stdout, stderr = self._run_command([
                    "git", "branch", "-d", branch
                ])
                
                if code == 0:
                    print(f"Deleted branch: {branch}")
                    deleted_branches.append(branch)
                else:
                    print(f"Failed to delete branch {branch}: {stderr}")
        
        return deleted_branches
    
    def cleanup_remote_branches(self) -> List[str]:
        """Clean up remote tracking branches that no longer exist."""
        print("Cleaning up remote tracking branches...")
        
        if self.dry_run:
            print("DRY RUN: Would prune remote branches")
            return []
        
        # Prune remote references
        code, stdout, stderr = self._run_command([
            "git", "remote", "prune", "origin"
        ])
        
        if code != 0:
            print(f"Failed to prune remote branches: {stderr}")
            return []
        
        # Parse output to get pruned branches
        pruned_branches = []
        for line in stdout.split('\n'):
            if 'pruned' in line.lower():
                # Extract branch name from output
                match = re.search(r'origin/(.+)', line)
                if match:
                    pruned_branches.append(match.group(1))
        
        if pruned_branches:
            print(f"Pruned {len(pruned_branches)} remote branches")
        else:
            print("No remote branches to prune")
        
        return pruned_branches
    
    def update_copyright_years(self) -> int:
        """Update copyright years in source files."""
        print("Updating copyright years...")
        
        current_year = datetime.now().year
        updated_files = 0
        
        # Find files that might contain copyright notices
        patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.md", "**/LICENSE"]
        
        for pattern in patterns:
            for file_path in self.repo_path.glob(pattern):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts[1:]):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Look for copyright patterns
                        copyright_pattern = r'(Copyright\s+\(c\)\s+)(\d{4})(-\d{4})?'
                        matches = re.finditer(copyright_pattern, content, re.IGNORECASE)
                        
                        updated_content = content
                        for match in matches:
                            prefix = match.group(1)
                            start_year = match.group(2)
                            end_year_group = match.group(3)
                            
                            if end_year_group:
                                end_year = int(end_year_group[1:])  # Remove the dash
                            else:
                                end_year = int(start_year)
                            
                            if end_year < current_year:
                                new_copyright = f"{prefix}{start_year}-{current_year}"
                                updated_content = updated_content.replace(match.group(0), new_copyright)
                        
                        if updated_content != content:
                            if self.dry_run:
                                print(f"DRY RUN: Would update copyright in {file_path}")
                                updated_files += 1
                            else:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(updated_content)
                                print(f"Updated copyright in {file_path}")
                                updated_files += 1
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return updated_files
    
    def check_large_files(self, size_limit_mb: float = 10.0) -> List[Tuple[str, float]]:
        """Check for large files in the repository."""
        print(f"Checking for files larger than {size_limit_mb}MB...")
        
        large_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        # Use git ls-files to get tracked files
        code, stdout, stderr = self._run_command([
            "git", "ls-files"
        ])
        
        if code != 0:
            print(f"Failed to get tracked files: {stderr}")
            return large_files
        
        for file_path in stdout.strip().split('\n'):
            if file_path:
                full_path = self.repo_path / file_path
                if full_path.exists():
                    try:
                        file_size = full_path.stat().st_size
                        if file_size > size_limit_bytes:
                            size_mb = file_size / (1024 * 1024)
                            large_files.append((file_path, size_mb))
                    except OSError:
                        continue
        
        if large_files:
            print(f"Found {len(large_files)} large files:")
            for file_path, size_mb in large_files:
                print(f"  {file_path}: {size_mb:.1f}MB")
        else:
            print("No large files found")
        
        return large_files
    
    def optimize_git_repository(self) -> bool:
        """Optimize git repository by running gc and cleaning up."""
        print("Optimizing git repository...")
        
        if self.dry_run:
            print("DRY RUN: Would optimize git repository")
            return True
        
        # Git garbage collection
        code, stdout, stderr = self._run_command([
            "git", "gc", "--aggressive", "--prune=now"
        ])
        
        if code != 0:
            print(f"Git gc failed: {stderr}")
            return False
        
        print("Git repository optimized")
        return True
    
    def update_gitignore_patterns(self) -> bool:
        """Update .gitignore with common patterns."""
        print("Checking .gitignore patterns...")
        
        gitignore_path = self.repo_path / ".gitignore"
        
        # Common patterns to ensure are present
        recommended_patterns = [
            "# Python",
            "__pycache__/",
            "*.py[cod]",
            "*.so",
            ".Python",
            "env/",
            "venv/",
            ".venv/",
            ".coverage",
            "htmlcov/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            "",
            "# IDE",
            ".vscode/settings.json",
            ".idea/",
            "*.swp",
            "*.swo",
            "",
            "# OS",
            ".DS_Store",
            "Thumbs.db",
            "",
            "# Logs",
            "*.log",
            "logs/",
            "",
            "# Temporary files",
            "*.tmp",
            "*.bak",
            "",
            "# Environment variables",
            ".env",
            ".env.local",
            ".env.*.local",
            "",
            "# Build artifacts", 
            "build/",
            "dist/",
            "*.egg-info/",
        ]
        
        if not gitignore_path.exists():
            if self.dry_run:
                print("DRY RUN: Would create .gitignore file")
                return True
            else:
                with open(gitignore_path, 'w') as f:
                    f.write('\n'.join(recommended_patterns))
                print("Created .gitignore file")
                return True
        
        # Read existing .gitignore
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
        
        # Check which patterns are missing
        missing_patterns = []
        for pattern in recommended_patterns:
            if pattern and pattern not in existing_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            if self.dry_run:
                print(f"DRY RUN: Would add {len(missing_patterns)} patterns to .gitignore")
                return True
            else:
                with open(gitignore_path, 'a') as f:
                    f.write('\n\n# Added by maintenance script\n')
                    f.write('\n'.join(missing_patterns))
                print(f"Added {len(missing_patterns)} patterns to .gitignore")
                return True
        else:
            print(".gitignore is up to date")
            return True
    
    def check_readme_freshness(self) -> Dict[str, any]:
        """Check if README needs updates."""
        print("Checking README freshness...")
        
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            return {"exists": False, "needs_update": True, "issues": ["README.md does not exist"]}
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for outdated installation instructions
        if "pip install" in content and "poetry install" not in content:
            issues.append("Consider adding Poetry installation instructions")
        
        # Check for missing badges
        if "![" not in content:
            issues.append("Consider adding status badges")
        
        # Check for TOC if content is long
        if len(content) > 3000 and "table of contents" not in content.lower():
            issues.append("Consider adding a table of contents")
        
        # Check for recent updates (look at git log)
        code, stdout, stderr = self._run_command([
            "git", "log", "-1", "--format=%ci", "README.md"
        ])
        
        last_updated = None
        if code == 0 and stdout:
            try:
                last_updated = datetime.fromisoformat(stdout.strip().split('+')[0])
                days_old = (datetime.now() - last_updated).days
                if days_old > 90:
                    issues.append(f"README last updated {days_old} days ago")
            except ValueError:
                pass
        
        return {
            "exists": True,
            "needs_update": len(issues) > 0,
            "issues": issues,
            "last_updated": last_updated
        }
    
    def generate_maintenance_report(self) -> str:
        """Generate comprehensive maintenance report."""
        print("Generating maintenance report...")
        
        # Collect all maintenance information
        readme_status = self.check_readme_freshness()
        large_files = self.check_large_files()
        
        # Get git statistics
        code, stdout, stderr = self._run_command([
            "git", "rev-list", "--count", "HEAD"
        ])
        total_commits = stdout.strip() if code == 0 else "Unknown"
        
        code, stdout, stderr = self._run_command([
            "git", "shortlog", "-sn"
        ])
        contributors = len(stdout.strip().split('\n')) if code == 0 and stdout else 0
        
        # Get repository size
        code, stdout, stderr = self._run_command([
            "git", "count-objects", "-vH"
        ])
        repo_size = "Unknown"
        if code == 0:
            for line in stdout.split('\n'):
                if line.startswith('size-pack'):
                    repo_size = line.split()[1]
                    break
        
        report = f"""# Repository Maintenance Report

**Generated**: {datetime.now().isoformat()}

## Repository Statistics

- **Total Commits**: {total_commits}
- **Contributors**: {contributors}
- **Repository Size**: {repo_size}

## Health Checks

### README Status
- **Exists**: {"✅" if readme_status["exists"] else "❌"}
- **Needs Update**: {"⚠️ Yes" if readme_status["needs_update"] else "✅ No"}

"""
        
        if readme_status["issues"]:
            report += "**Issues Found:**\n"
            for issue in readme_status["issues"]:
                report += f"- {issue}\n"
        
        report += "\n### Large Files\n"
        if large_files:
            report += f"Found {len(large_files)} files larger than 10MB:\n\n"
            for file_path, size_mb in large_files:
                report += f"- `{file_path}`: {size_mb:.1f}MB\n"
        else:
            report += "✅ No large files found\n"
        
        report += """
## Maintenance Actions Available

- 🧹 **Branch Cleanup**: Remove old merged branches
- 🔄 **Remote Cleanup**: Prune deleted remote branches  
- 📅 **Copyright Update**: Update copyright years
- 🗜️ **Git Optimization**: Run garbage collection
- 📝 **Gitignore Update**: Add recommended patterns

## Next Steps

1. Review large files and consider using Git LFS
2. Update README if issues were found
3. Run maintenance actions as needed

---

*This report was generated automatically by the repository maintenance script.*
"""
        
        return report
    
    def run_full_maintenance(self, keep_branches_days: int = 7) -> Dict[str, any]:
        """Run all maintenance tasks."""
        print("Running full repository maintenance...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "tasks": {}
        }
        
        # Clean up branches
        try:
            deleted_branches = self.cleanup_merged_branches(keep_branches_days)
            pruned_remotes = self.cleanup_remote_branches()
            results["tasks"]["branch_cleanup"] = {
                "success": True,
                "deleted_local": deleted_branches,
                "pruned_remote": pruned_remotes
            }
        except Exception as e:
            results["tasks"]["branch_cleanup"] = {
                "success": False,
                "error": str(e)
            }
        
        # Update copyright years
        try:
            updated_files = self.update_copyright_years()
            results["tasks"]["copyright_update"] = {
                "success": True,
                "updated_files": updated_files
            }
        except Exception as e:
            results["tasks"]["copyright_update"] = {
                "success": False,
                "error": str(e)
            }
        
        # Update .gitignore
        try:
            gitignore_updated = self.update_gitignore_patterns()
            results["tasks"]["gitignore_update"] = {
                "success": gitignore_updated
            }
        except Exception as e:
            results["tasks"]["gitignore_update"] = {
                "success": False,
                "error": str(e)
            }
        
        # Optimize git repository
        try:
            optimized = self.optimize_git_repository()
            results["tasks"]["git_optimization"] = {
                "success": optimized
            }
        except Exception as e:
            results["tasks"]["git_optimization"] = {
                "success": False,
                "error": str(e)
            }
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--keep-branches-days", type=int, default=7, help="Keep merged branches newer than this many days")
    parser.add_argument("--task", choices=["branches", "copyright", "gitignore", "optimize", "report", "full"], 
                       help="Run specific maintenance task")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    maintainer = RepositoryMaintainer(args.repo_path, args.dry_run)
    
    if args.task == "branches":
        deleted = maintainer.cleanup_merged_branches(args.keep_branches_days)
        pruned = maintainer.cleanup_remote_branches()
        print(f"Maintenance completed: {len(deleted)} local branches deleted, {len(pruned)} remote references pruned")
    
    elif args.task == "copyright":
        updated = maintainer.update_copyright_years()
        print(f"Copyright maintenance completed: {updated} files updated")
    
    elif args.task == "gitignore":
        updated = maintainer.update_gitignore_patterns()
        print(f"Gitignore maintenance completed: {'updated' if updated else 'no changes needed'}")
    
    elif args.task == "optimize":
        success = maintainer.optimize_git_repository()
        print(f"Git optimization completed: {'success' if success else 'failed'}")
    
    elif args.task == "report":
        report = maintainer.generate_maintenance_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    elif args.task == "full" or args.task is None:
        results = maintainer.run_full_maintenance(args.keep_branches_days)
        
        print("\n=== Maintenance Summary ===")
        for task_name, task_result in results["tasks"].items():
            status = "✅" if task_result["success"] else "❌"
            print(f"{status} {task_name.replace('_', ' ').title()}")
            if not task_result["success"] and "error" in task_result:
                print(f"   Error: {task_result['error']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to {args.output}")
    
    print("Repository maintenance completed!")


if __name__ == "__main__":
    main()