#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuous SDLC value identification and prioritization system.
"""

import json
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
# import yaml  # Not available in this environment
import json as yaml_replacement


class ValueCategory(Enum):
    SECURITY = "security"
    TECHNICAL_DEBT = "technical_debt"  
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    INFRASTRUCTURE = "infrastructure"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValueItem:
    id: str
    title: str
    description: str
    category: ValueCategory
    priority: Priority
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Effort estimation
    estimated_hours: float = 0.0
    confidence: float = 0.0
    
    # Context
    files_affected: List[str] = None
    dependencies: List[str] = None
    
    # Tracking
    discovered_at: str = ""
    last_updated: str = ""
    source: str = ""
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []
        if self.dependencies is None:
            self.dependencies = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.discovered_at


class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization engine."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.value_items: List[ValueItem] = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        if self.config_path.exists() and str(self.config_path).endswith('.json'):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        # Return default config for YAML files or if not found
        return {
            'scoring': {
                'weights': {
                    'advanced': {
                        'wsjf': 0.5,
                        'ice': 0.1,
                        'technicalDebt': 0.3,
                        'security': 0.1
                    }
                },
                'thresholds': {
                    'minScore': 15,
                    'maxRisk': 0.8,
                    'securityBoost': 2.0,
                    'complianceBoost': 1.8
                }
            }
        }
    
    def discover_value_opportunities(self) -> List[ValueItem]:
        """Execute comprehensive value discovery across all sources."""
        print("🔍 Starting autonomous value discovery...")
        
        self.value_items = []
        
        # Parallel discovery from multiple sources
        discovery_methods = [
            self._discover_from_git_history,
            self._discover_from_static_analysis,
            self._discover_from_security_scans,
            self._discover_from_dependency_audit,
            self._discover_from_test_coverage,
            self._discover_from_performance_profiling,
            self._discover_from_quantum_optimization,
            self._discover_from_documentation_gaps,
            self._discover_from_infrastructure_analysis,
        ]
        
        for method in discovery_methods:
            try:
                method()
            except Exception as e:
                print(f"⚠️  Discovery method {method.__name__} failed: {e}")
        
        # Score and prioritize all discovered items
        self._calculate_composite_scores()
        self._rank_by_value()
        
        print(f"✅ Discovered {len(self.value_items)} value opportunities")
        return self.value_items
    
    def _discover_from_git_history(self):
        """Extract value opportunities from Git history."""
        print("📚 Analyzing Git history for TODO/FIXME markers...")
        
        # Search for technical debt markers in commit messages and code
        patterns = [
            (r'TODO:', 'Code cleanup needed'),
            (r'FIXME:', 'Bug fix required'),
            (r'XXX:', 'Critical issue'),
            (r'HACK:', 'Temporary solution needs refactoring'),
            (r'DEPRECATED:', 'Legacy code removal'),
            (r'OPTIMIZE:', 'Performance improvement needed'),
        ]
        
        for pattern, description in patterns:
            cmd = f"git grep -n '{pattern}' -- '*.py' '*.md' '*.yaml' '*.yml' || true"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    self._create_technical_debt_item(line, description)
    
    def _discover_from_static_analysis(self):
        """Run static analysis tools to find code quality issues."""
        print("🔍 Running static analysis for code quality issues...")
        
        # Run ruff for linting issues
        try:
            result = subprocess.run(['ruff', 'check', '.', '--format=json'], 
                                  capture_output=True, text=True)
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues:
                    self._create_quality_item_from_ruff(issue)
        except Exception as e:
            print(f"⚠️  Ruff analysis failed: {e}")
        
        # Run mypy for type checking issues
        try:
            result = subprocess.run(['mypy', 'src/', '--json-report', '/tmp/mypy-report'],
                                  capture_output=True, text=True, cwd=self.repo_root)
            # Parse mypy results and create type safety items
            self._parse_mypy_results()
        except Exception as e:
            print(f"⚠️  MyPy analysis failed: {e}")
    
    def _discover_from_security_scans(self):
        """Identify security vulnerabilities and improvements."""
        print("🔒 Scanning for security vulnerabilities...")
        
        # Run safety check for known vulnerabilities
        try:
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True)
            if result.stdout:
                safety_report = json.loads(result.stdout)
                for vuln in safety_report:
                    self._create_security_item(vuln)
        except Exception as e:
            print(f"⚠️  Safety scan failed: {e}")
        
        # Run bandit for security issues in code
        try:
            result = subprocess.run(['bandit', '-r', 'src/', '-f', 'json'], 
                                  capture_output=True, text=True)
            if result.stdout:
                bandit_report = json.loads(result.stdout)
                for issue in bandit_report.get('results', []):
                    self._create_security_code_item(issue)
        except Exception as e:
            print(f"⚠️  Bandit scan failed: {e}")
    
    def _discover_from_dependency_audit(self):
        """Analyze dependencies for updates and security issues."""
        print("📦 Auditing dependencies for updates and security...")
        
        # Check for outdated packages using poetry
        try:
            result = subprocess.run(['poetry', 'show', '--outdated'], 
                                  capture_output=True, text=True)
            if result.stdout:
                self._parse_outdated_dependencies(result.stdout)
        except Exception as e:
            print(f"⚠️  Dependency audit failed: {e}")
    
    def _discover_from_test_coverage(self):
        """Identify testing gaps and coverage improvements."""
        print("🧪 Analyzing test coverage for improvement opportunities...")
        
        # Run coverage analysis
        try:
            subprocess.run(['coverage', 'run', '-m', 'pytest'], 
                          capture_output=True, cwd=self.repo_root)
            result = subprocess.run(['coverage', 'json'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            
            if Path('coverage.json').exists():
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                self._analyze_coverage_gaps(coverage_data)
        except Exception as e:
            print(f"⚠️  Coverage analysis failed: {e}")
    
    def _discover_from_performance_profiling(self):
        """Identify performance bottlenecks and optimization opportunities."""
        print("⚡ Profiling for performance optimization opportunities...")
        
        # Analyze large/complex files that might benefit from optimization
        for py_file in self.repo_root.rglob('*.py'):
            if py_file.stat().st_size > 5000:  # Files larger than 5KB
                lines = len(py_file.read_text().splitlines())
                if lines > 200:  # Large files
                    self.value_items.append(ValueItem(
                        id=f"perf-{py_file.stem}",
                        title=f"Optimize large file {py_file.name}",
                        description=f"File has {lines} lines, consider refactoring for performance",
                        category=ValueCategory.PERFORMANCE,
                        priority=Priority.MEDIUM,
                        files_affected=[str(py_file)],
                        estimated_hours=3.0,
                        confidence=0.7,
                        source="performance_profiling"
                    ))
    
    def _discover_from_quantum_optimization(self):
        """Identify quantum-specific optimization opportunities."""
        print("⚛️  Analyzing quantum computation optimization opportunities...")
        
        # Look for quantum backend usage patterns that could be optimized
        quantum_files = list(self.repo_root.rglob('*quantum*.py')) + \
                       list(self.repo_root.rglob('*backend*.py'))
        
        for qfile in quantum_files:
            try:
                content = qfile.read_text()
                
                # Check for potential quantum optimizations
                if 'qiskit' in content and 'optimize' not in content.lower():
                    self.value_items.append(ValueItem(
                        id=f"quantum-opt-{qfile.stem}",
                        title=f"Add quantum circuit optimization to {qfile.name}",
                        description="Implement quantum circuit optimization for better performance",
                        category=ValueCategory.QUANTUM_OPTIMIZATION,
                        priority=Priority.HIGH,
                        files_affected=[str(qfile)],
                        estimated_hours=6.0,
                        confidence=0.8,
                        source="quantum_optimization"
                    ))
                
                if 'backend' in content.lower() and 'fallback' not in content.lower():
                    self.value_items.append(ValueItem(
                        id=f"quantum-fallback-{qfile.stem}",
                        title=f"Add classical fallback to {qfile.name}",
                        description="Implement classical fallback for quantum backend failures",
                        category=ValueCategory.QUANTUM_OPTIMIZATION,
                        priority=Priority.MEDIUM,
                        files_affected=[str(qfile)],
                        estimated_hours=4.0,
                        confidence=0.9,
                        source="quantum_optimization"
                    ))
            except Exception as e:
                continue
    
    def _discover_from_documentation_gaps(self):
        """Identify documentation improvements needed."""
        print("📖 Analyzing documentation gaps...")
        
        # Check for Python files without docstrings
        for py_file in self.repo_root.rglob('src/**/*.py'):
            try:
                content = py_file.read_text()
                if 'def ' in content or 'class ' in content:
                    if '"""' not in content and "'''" not in content:
                        self.value_items.append(ValueItem(
                            id=f"doc-{py_file.stem}",
                            title=f"Add docstrings to {py_file.name}",
                            description="Module lacks proper documentation",
                            category=ValueCategory.DOCUMENTATION,
                            priority=Priority.LOW,
                            files_affected=[str(py_file)],
                            estimated_hours=1.5,
                            confidence=0.9,
                            source="documentation_gaps"
                        ))
            except Exception:
                continue
    
    def _discover_from_infrastructure_analysis(self):
        """Analyze infrastructure and deployment optimizations."""
        print("🏗️  Analyzing infrastructure optimization opportunities...")
        
        # Check for missing GitHub workflows
        workflows_dir = Path('.github/workflows')
        if not workflows_dir.exists():
            self.value_items.append(ValueItem(
                id="infra-github-workflows",
                title="Deploy GitHub Actions workflows",
                description="Copy workflow templates to .github/workflows/ for CI/CD automation",
                category=ValueCategory.INFRASTRUCTURE,
                priority=Priority.HIGH,
                estimated_hours=2.0,
                confidence=0.95,
                source="infrastructure_analysis"
            ))
        
        # Check Docker optimization opportunities
        dockerfile = Path('Dockerfile')
        if dockerfile.exists():
            content = dockerfile.read_text()
            if 'multi-stage' not in content.lower():
                self.value_items.append(ValueItem(
                    id="infra-docker-multistage",
                    title="Implement multi-stage Docker build",
                    description="Optimize Docker image size with multi-stage builds",
                    category=ValueCategory.INFRASTRUCTURE,
                    priority=Priority.MEDIUM,
                    files_affected=['Dockerfile'],
                    estimated_hours=3.0,
                    confidence=0.8,
                    source="infrastructure_analysis"
                ))
    
    def _calculate_composite_scores(self):
        """Calculate composite scores for all value items using WSJF + ICE + Technical Debt."""
        weights = self.config.get('scoring', {}).get('weights', {}).get('advanced', {})
        
        for item in self.value_items:
            # WSJF Calculation (Weighted Shortest Job First)
            user_value = self._calculate_user_business_value(item)
            time_criticality = self._calculate_time_criticality(item)
            risk_reduction = self._calculate_risk_reduction(item)
            opportunity_enablement = self._calculate_opportunity_enablement(item)
            
            cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
            job_size = max(item.estimated_hours, 0.5)  # Minimum 0.5 hours
            item.wsjf_score = cost_of_delay / job_size
            
            # ICE Calculation (Impact, Confidence, Ease)
            impact = self._calculate_impact(item)
            confidence_val = item.confidence or 0.5
            ease = 10 - min(item.estimated_hours, 10)  # Easier if takes less time
            item.ice_score = impact * confidence_val * ease
            
            # Technical Debt Score
            item.technical_debt_score = self._calculate_technical_debt_score(item)
            
            # Composite Score with adaptive weighting
            item.composite_score = (
                weights.get('wsjf', 0.5) * self._normalize_score(item.wsjf_score, 0, 50) +
                weights.get('ice', 0.1) * self._normalize_score(item.ice_score, 0, 1000) +
                weights.get('technicalDebt', 0.3) * self._normalize_score(item.technical_debt_score, 0, 100) +
                weights.get('security', 0.1) * (2.0 if item.category == ValueCategory.SECURITY else 1.0)
            )
            
            # Apply priority boosts
            if item.priority == Priority.CRITICAL:
                item.composite_score *= 2.0
            elif item.priority == Priority.HIGH:
                item.composite_score *= 1.5
    
    def _rank_by_value(self):
        """Sort value items by composite score (highest first)."""
        self.value_items.sort(key=lambda x: x.composite_score, reverse=True)
    
    def _calculate_user_business_value(self, item: ValueItem) -> float:
        """Calculate user/business value component."""
        category_values = {
            ValueCategory.SECURITY: 10.0,
            ValueCategory.PERFORMANCE: 8.0,
            ValueCategory.QUANTUM_OPTIMIZATION: 9.0,
            ValueCategory.TECHNICAL_DEBT: 6.0,
            ValueCategory.TESTING: 7.0,
            ValueCategory.DOCUMENTATION: 4.0,
            ValueCategory.DEPENDENCY: 5.0,
            ValueCategory.INFRASTRUCTURE: 7.0,
        }
        return category_values.get(item.category, 5.0)
    
    def _calculate_time_criticality(self, item: ValueItem) -> float:
        """Calculate time criticality component."""
        if item.category == ValueCategory.SECURITY:
            return 10.0
        elif item.priority == Priority.CRITICAL:
            return 8.0
        elif item.priority == Priority.HIGH:
            return 6.0
        return 3.0
    
    def _calculate_risk_reduction(self, item: ValueItem) -> float:
        """Calculate risk reduction value."""
        risk_categories = [ValueCategory.SECURITY, ValueCategory.TECHNICAL_DEBT]
        return 8.0 if item.category in risk_categories else 2.0
    
    def _calculate_opportunity_enablement(self, item: ValueItem) -> float:
        """Calculate opportunity enablement value."""
        enabling_categories = [ValueCategory.INFRASTRUCTURE, ValueCategory.QUANTUM_OPTIMIZATION]
        return 6.0 if item.category in enabling_categories else 2.0
    
    def _calculate_impact(self, item: ValueItem) -> float:
        """Calculate impact score (1-10)."""
        impact_map = {
            ValueCategory.SECURITY: 9.0,
            ValueCategory.PERFORMANCE: 8.0,
            ValueCategory.QUANTUM_OPTIMIZATION: 8.5,
            ValueCategory.INFRASTRUCTURE: 7.0,
            ValueCategory.TECHNICAL_DEBT: 6.0,
            ValueCategory.TESTING: 6.5,
            ValueCategory.DEPENDENCY: 5.0,
            ValueCategory.DOCUMENTATION: 4.0,
        }
        return impact_map.get(item.category, 5.0)
    
    def _calculate_technical_debt_score(self, item: ValueItem) -> float:
        """Calculate technical debt reduction score."""
        if item.category == ValueCategory.TECHNICAL_DEBT:
            return min(item.estimated_hours * 10, 100)
        elif item.category in [ValueCategory.SECURITY, ValueCategory.PERFORMANCE]:
            return min(item.estimated_hours * 5, 50)
        return min(item.estimated_hours * 2, 20)
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 50.0
        return max(0, min(100, ((score - min_val) / (max_val - min_val)) * 100))
    
    # Helper methods for parsing tool outputs
    def _create_technical_debt_item(self, line: str, description: str):
        """Create technical debt item from git grep result."""
        parts = line.split(':', 2)
        if len(parts) >= 2:
            filename, line_num = parts[0], parts[1]
            self.value_items.append(ValueItem(
                id=f"td-{len(self.value_items)}",
                title=f"Technical debt in {Path(filename).name}:{line_num}",
                description=description,
                category=ValueCategory.TECHNICAL_DEBT,
                priority=Priority.MEDIUM,
                files_affected=[filename],
                estimated_hours=2.0,
                confidence=0.8,
                source="git_history"
            ))
    
    def _create_quality_item_from_ruff(self, issue: Dict):
        """Create quality item from ruff issue."""
        self.value_items.append(ValueItem(
            id=f"quality-{len(self.value_items)}",
            title=f"Fix {issue.get('code', 'linting')} issue in {Path(issue['filename']).name}",
            description=issue.get('message', 'Code quality issue'),
            category=ValueCategory.TECHNICAL_DEBT,
            priority=Priority.LOW,
            files_affected=[issue['filename']],
            estimated_hours=0.5,
            confidence=0.9,
            source="static_analysis"
        ))
    
    def _create_security_item(self, vuln: Dict):
        """Create security item from safety vulnerability."""
        self.value_items.append(ValueItem(
            id=f"sec-{len(self.value_items)}",
            title=f"Update {vuln.get('package', 'dependency')} for security",
            description=f"Security vulnerability: {vuln.get('advisory', 'Unknown')}",
            category=ValueCategory.SECURITY,
            priority=Priority.HIGH,
            estimated_hours=1.0,
            confidence=0.95,
            source="security_scan"
        ))
    
    def save_backlog(self, filename: str = ".terragon/backlog.json"):
        """Save prioritized backlog to file."""
        # Convert items to JSON-serializable format
        items_json = []
        for item in self.value_items:
            item_dict = asdict(item)
            item_dict['category'] = item.category.value
            item_dict['priority'] = item.priority.value
            items_json.append(item_dict)
        
        backlog_data = {
            "generated_at": datetime.now().isoformat(),
            "repository": "quantum-agent-scheduler",
            "total_items": len(self.value_items),
            "items": items_json
        }
        
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(backlog_data, f, indent=2)
        
        print(f"💾 Saved {len(self.value_items)} items to {filename}")
    
    def generate_backlog_markdown(self) -> str:
        """Generate markdown backlog report."""
        top_items = self.value_items[:10]
        
        report = f"""# 📊 Autonomous Value Backlog

**Repository**: quantum-agent-scheduler  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Items Discovered**: {len(self.value_items)}

## 🎯 Next Best Value Item

"""
        
        if top_items:
            item = top_items[0]
            report += f"""**[{item.id.upper()}] {item.title}**
- **Composite Score**: {item.composite_score:.1f}
- **WSJF**: {item.wsjf_score:.1f} | **ICE**: {item.ice_score:.0f} | **Tech Debt**: {item.technical_debt_score:.1f}
- **Category**: {item.category.value}
- **Estimated Effort**: {item.estimated_hours:.1f} hours
- **Expected Impact**: {item.description}

"""
        
        report += """## 📋 Top 10 Value Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(top_items, 1):
            report += f"| {i} | {item.id} | {item.title[:50]}... | {item.composite_score:.1f} | {item.category.value} | {item.estimated_hours:.1f} |\n"
        
        report += f"""

## 📈 Discovery Summary

**Items by Category**:
"""
        
        category_counts = {}
        for item in self.value_items:
            category_counts[item.category.value] = category_counts.get(item.category.value, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            report += f"- **{category.replace('_', ' ').title()}**: {count} items\n"
        
        report += f"""
**Discovery Sources**:
"""
        
        source_counts = {}
        for item in self.value_items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        
        for source, count in sorted(source_counts.items()):
            percentage = (count / len(self.value_items)) * 100
            report += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        report += """
## 🔄 Continuous Execution

This backlog is continuously updated through automated discovery. The top-scored items represent the highest value opportunities for autonomous execution.

**Next Update**: Hourly security scans, Daily comprehensive analysis
"""
        
        return report


def main():
    """Main execution function for autonomous value discovery."""
    engine = ValueDiscoveryEngine()
    
    # Discover all value opportunities
    value_items = engine.discover_value_opportunities()
    
    # Save results
    engine.save_backlog()
    
    # Generate markdown report
    markdown_report = engine.generate_backlog_markdown()
    with open('AUTONOMOUS_BACKLOG.md', 'w') as f:
        f.write(markdown_report)
    
    print(f"✅ Value discovery complete!")
    print(f"   📊 {len(value_items)} opportunities discovered")
    print(f"   🎯 Top item: {value_items[0].title if value_items else 'None'}")
    print(f"   📄 Report: AUTONOMOUS_BACKLOG.md")
    
    return value_items


if __name__ == "__main__":
    main()