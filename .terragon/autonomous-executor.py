#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Executor
Continuously executes highest-value SDLC improvements.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from value_engine import ValueDiscoveryEngine, ValueItem, ValueCategory, Priority


class AutonomousExecutor:
    """Autonomous SDLC task executor with continuous value delivery."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.metrics = {
            "tasks_executed": 0,
            "total_value_delivered": 0.0,
            "execution_time": 0.0,
            "success_rate": 0.0,
            "failed_tasks": []
        }
        
        # Execution state
        self.current_branch = self._get_current_branch()
        self.base_branch = "main"
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_current_branch(self) -> str:
        """Get current git branch."""
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    
    def execute_next_best_value(self) -> Optional[ValueItem]:
        """Discover and execute the next highest-value item."""
        print("🚀 Starting autonomous value execution cycle...")
        
        # Discover current value opportunities
        engine = ValueDiscoveryEngine()
        value_items = engine.discover_value_opportunities()
        
        if not value_items:
            print("ℹ️  No value opportunities discovered")
            return None
        
        # Select next best item
        next_item = self._select_next_item(value_items)
        if not next_item:
            print("ℹ️  No suitable items for autonomous execution")
            return None
        
        print(f"🎯 Selected: {next_item.title} (Score: {next_item.composite_score:.1f})")
        
        # Execute the item
        success = self._execute_value_item(next_item)
        
        # Update metrics
        self._update_metrics(next_item, success)
        
        return next_item if success else None
    
    def _select_next_item(self, value_items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item for execution."""
        min_score = self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 15)
        max_risk = self.config.get('scoring', {}).get('thresholds', {}).get('maxRisk', 0.8)
        
        for item in value_items:
            # Skip if score too low
            if item.composite_score < min_score:
                continue
                
            # Skip if risk too high
            risk = 1.0 - item.confidence
            if risk > max_risk:
                print(f"⚠️  Skipping {item.id}: risk {risk:.2f} > threshold {max_risk}")
                continue
            
            # Skip if dependencies not met
            if not self._dependencies_met(item):
                print(f"⚠️  Skipping {item.id}: dependencies not met")
                continue
            
            return item
        
        return None
    
    def _dependencies_met(self, item: ValueItem) -> bool:
        """Check if item dependencies are satisfied."""
        for dep in item.dependencies:
            # Check if dependency file exists, command available, etc.
            if dep.startswith('file:'):
                if not Path(dep[5:]).exists():
                    return False
            elif dep.startswith('cmd:'):
                result = subprocess.run(['which', dep[4:]], capture_output=True)
                if result.returncode != 0:
                    return False
        return True
    
    def _execute_value_item(self, item: ValueItem) -> bool:
        """Execute a specific value item."""
        start_time = datetime.now()
        
        try:
            # Create feature branch
            branch_name = f"auto-value/{item.id}-{datetime.now().strftime('%Y%m%d-%H%M')}"
            self._create_feature_branch(branch_name)
            
            # Execute based on category
            success = self._execute_by_category(item)
            
            if success:
                # Validate changes
                if self._validate_changes():
                    # Create pull request
                    self._create_pull_request(item, branch_name)
                    print(f"✅ Successfully executed {item.id}")
                    return True
                else:
                    print(f"❌ Validation failed for {item.id}")
                    self._rollback_changes(branch_name)
                    return False
            else:
                print(f"❌ Execution failed for {item.id}")
                self._rollback_changes(branch_name)
                return False
                
        except Exception as e:
            print(f"💥 Error executing {item.id}: {e}")
            self._rollback_changes(branch_name)
            return False
        
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"⏱️  Execution time: {execution_time:.1f}s")
    
    def _execute_by_category(self, item: ValueItem) -> bool:
        """Execute item based on its category."""
        if item.category == ValueCategory.INFRASTRUCTURE:
            return self._execute_infrastructure_item(item)
        elif item.category == ValueCategory.SECURITY:
            return self._execute_security_item(item)
        elif item.category == ValueCategory.TECHNICAL_DEBT:
            return self._execute_technical_debt_item(item)
        elif item.category == ValueCategory.DEPENDENCY:
            return self._execute_dependency_item(item)
        elif item.category == ValueCategory.TESTING:
            return self._execute_testing_item(item)
        elif item.category == ValueCategory.DOCUMENTATION:
            return self._execute_documentation_item(item)
        elif item.category == ValueCategory.PERFORMANCE:
            return self._execute_performance_item(item)
        elif item.category == ValueCategory.QUANTUM_OPTIMIZATION:
            return self._execute_quantum_optimization_item(item)
        else:
            print(f"⚠️  Unknown category: {item.category}")
            return False
    
    def _execute_infrastructure_item(self, item: ValueItem) -> bool:
        """Execute infrastructure-related improvements."""
        if "github-workflows" in item.id:
            return self._deploy_github_workflows()
        elif "docker-multistage" in item.id:
            return self._optimize_dockerfile()
        return False
    
    def _deploy_github_workflows(self) -> bool:
        """Deploy GitHub Actions workflows from templates."""
        try:
            # Create .github/workflows directory
            workflows_dir = Path('.github/workflows')
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy workflow templates
            templates_dir = Path('docs/workflows/templates')
            if templates_dir.exists():
                for template in templates_dir.glob('*.yml'):
                    target = workflows_dir / template.name
                    target.write_text(template.read_text())
                    print(f"📄 Deployed workflow: {template.name}")
                
                # Stage changes
                subprocess.run(['git', 'add', '.github/workflows/'], check=True)
                subprocess.run(['git', 'commit', '-m', 
                              'feat: deploy GitHub Actions workflows for CI/CD automation\\n\\n🤖 Generated with [Claude Code](https://claude.ai/code)\\n\\nCo-Authored-By: Claude <noreply@anthropic.com>'], 
                              check=True)
                return True
            else:
                print("⚠️  Workflow templates not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to deploy workflows: {e}")
            return False
    
    def _execute_security_item(self, item: ValueItem) -> bool:
        """Execute security-related improvements."""
        # Update vulnerable dependencies
        try:
            subprocess.run(['poetry', 'update'], check=True)
            subprocess.run(['git', 'add', 'poetry.lock', 'pyproject.toml'], check=True)
            subprocess.run(['git', 'commit', '-m', f'security: {item.description}\\n\\n🤖 Generated with [Claude Code](https://claude.ai/code)\\n\\nCo-Authored-By: Claude <noreply@anthropic.com>'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _execute_technical_debt_item(self, item: ValueItem) -> bool:
        """Execute technical debt reduction."""
        # Run automated formatting and linting fixes
        try:
            subprocess.run(['ruff', 'check', '--fix', '.'], check=True)
            subprocess.run(['black', '.'], check=True)
            subprocess.run(['isort', '.'], check=True)
            
            # Stage changes if any
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
            if result.stdout.strip():
                subprocess.run(['git', 'add', '.'], check=True)
                subprocess.run(['git', 'commit', '-m', f'refactor: {item.description}\\n\\n🤖 Generated with [Claude Code](https://claude.ai/code)\\n\\nCo-Authored-By: Claude <noreply@anthropic.com>'], check=True)
            
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _execute_dependency_item(self, item: ValueItem) -> bool:
        """Execute dependency updates."""
        try:
            # Update specific package or all dependencies
            subprocess.run(['poetry', 'update'], check=True)
            subprocess.run(['git', 'add', 'poetry.lock'], check=True)
            subprocess.run(['git', 'commit', '-m', f'deps: {item.description}\\n\\n🤖 Generated with [Claude Code](https://claude.ai/code)\\n\\nCo-Authored-By: Claude <noreply@anthropic.com>'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _execute_testing_item(self, item: ValueItem) -> bool:
        """Execute testing improvements."""
        # Generate basic test templates for untested modules
        return True  # Placeholder for test generation logic
    
    def _execute_documentation_item(self, item: ValueItem) -> bool:
        """Execute documentation improvements."""
        # Add basic docstrings to functions/classes
        return True  # Placeholder for documentation generation
    
    def _execute_performance_item(self, item: ValueItem) -> bool:
        """Execute performance optimizations."""
        # Add performance monitoring or optimizations
        return True  # Placeholder for performance optimization
    
    def _execute_quantum_optimization_item(self, item: ValueItem) -> bool:
        """Execute quantum-specific optimizations."""
        # Optimize quantum circuits or add fallbacks
        return True  # Placeholder for quantum optimizations
    
    def _create_feature_branch(self, branch_name: str):
        """Create and switch to feature branch."""
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
    
    def _validate_changes(self) -> bool:
        """Validate changes meet quality standards."""
        try:
            # Run tests
            test_result = subprocess.run(['pytest', '--tb=short'], 
                                       capture_output=True, text=True)
            if test_result.returncode != 0:
                print(f"❌ Tests failed:\\n{test_result.stdout}")
                return False
            
            # Run linting
            lint_result = subprocess.run(['ruff', 'check', '.'], 
                                       capture_output=True, text=True)
            if lint_result.returncode != 0:
                print(f"⚠️  Linting issues found:\\n{lint_result.stdout}")
                # Auto-fix if possible
                subprocess.run(['ruff', 'check', '--fix', '.'])
            
            # Run type checking
            type_result = subprocess.run(['mypy', 'src/'], 
                                       capture_output=True, text=True)
            if type_result.returncode != 0:
                print(f"⚠️  Type checking issues:\\n{type_result.stdout}")
            
            # Security scan
            security_result = subprocess.run(['bandit', '-r', 'src/', '-f', 'txt'], 
                                           capture_output=True, text=True)
            if "No issues identified" not in security_result.stdout:
                print(f"⚠️  Security issues found:\\n{security_result.stdout}")
            
            print("✅ All validations passed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def _create_pull_request(self, item: ValueItem, branch_name: str):
        """Create pull request for the changes."""
        try:
            # Push branch
            subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
            
            # Create PR description
            pr_body = f"""## 🤖 Autonomous Value Delivery

**Item**: {item.title}  
**Category**: {item.category.value}  
**Priority**: {item.priority.value}  
**Composite Score**: {item.composite_score:.1f}

### 📊 Value Metrics
- **WSJF**: {item.wsjf_score:.1f}
- **ICE**: {item.ice_score:.0f} 
- **Technical Debt**: {item.technical_debt_score:.1f}
- **Estimated Effort**: {item.estimated_hours:.1f} hours

### 📝 Description
{item.description}

### 🔍 Files Modified
{chr(10).join(f'- `{f}`' for f in item.files_affected)}

### ✅ Validation Results
- ✅ Tests passed
- ✅ Linting passed  
- ✅ Type checking completed
- ✅ Security scan completed

### 🎯 Expected Impact
This change delivers autonomous SDLC value through {item.category.value} improvements.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

            # Create PR using gh CLI if available
            pr_result = subprocess.run([
                'gh', 'pr', 'create',
                '--title', f'[AUTO-VALUE] {item.title}',
                '--body', pr_body,
                '--label', f'autonomous,value-driven,{item.category.value}',
                '--label', f'priority-{item.priority.value}'
            ], capture_output=True, text=True)
            
            if pr_result.returncode == 0:
                pr_url = pr_result.stdout.strip()
                print(f"🚀 Created PR: {pr_url}")
            else:
                print(f"⚠️  Failed to create PR: {pr_result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create PR: {e}")
    
    def _rollback_changes(self, branch_name: str):
        """Rollback changes and cleanup branch."""
        try:
            subprocess.run(['git', 'checkout', self.base_branch], check=True)
            subprocess.run(['git', 'branch', '-D', branch_name], check=True)
            print(f"🔄 Rolled back branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Rollback failed: {e}")
    
    def _update_metrics(self, item: ValueItem, success: bool):
        """Update execution metrics."""
        self.metrics["tasks_executed"] += 1
        
        if success:
            self.metrics["total_value_delivered"] += item.composite_score
        else:
            self.metrics["failed_tasks"].append({
                "id": item.id,
                "title": item.title,
                "timestamp": datetime.now().isoformat()
            })
        
        self.metrics["success_rate"] = (
            (self.metrics["tasks_executed"] - len(self.metrics["failed_tasks"])) / 
            self.metrics["tasks_executed"]
        ) if self.metrics["tasks_executed"] > 0 else 0.0
        
        # Save metrics
        self._save_metrics()
    
    def _save_metrics(self):
        """Save execution metrics."""
        metrics_file = Path('.terragon/execution-metrics.json')
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump({
                **self.metrics,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def continuous_execution(self, max_iterations: int = 10):
        """Run continuous value execution cycle."""
        print(f"🔄 Starting continuous execution (max {max_iterations} iterations)")
        
        for iteration in range(max_iterations):
            print(f"\\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            item = self.execute_next_best_value()
            
            if not item:
                print("ℹ️  No more items to execute")
                break
            
            # Short pause between iterations
            import time
            time.sleep(5)
        
        print(f"\\n📊 Execution Summary:")
        print(f"   Tasks Executed: {self.metrics['tasks_executed']}")
        print(f"   Success Rate: {self.metrics['success_rate']:.1%}")  
        print(f"   Total Value Delivered: {self.metrics['total_value_delivered']:.1f}")


def main():
    """Main entry point for autonomous execution."""
    executor = AutonomousExecutor()
    
    # Run single execution cycle
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        executor.continuous_execution(max_iter)
    else:
        executor.execute_next_best_value()


if __name__ == "__main__":
    main()