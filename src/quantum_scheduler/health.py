"""Health checks and system diagnostics for quantum scheduler."""

import time
import logging
import platform
import sys
from typing import Dict, Any, List
from dataclasses import dataclass

from .core import QuantumScheduler, Agent, Task
from .backends import ClassicalBackend
from .monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "ok", "warning", "error"
    message: str
    duration: float = 0.0
    details: Dict[str, Any] = None


class HealthChecker:
    """Performs comprehensive health checks."""
    
    def __init__(self):
        self.checks = [
            self._check_system_info,
            self._check_dependencies,
            self._check_scheduler_basic,
            self._check_backend_availability,
            self._check_metrics_system,
            self._check_validation_system,
            self._check_performance
        ]
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks.
        
        Returns:
            List of health check results
        """
        results = []
        
        logger.info("Starting comprehensive health check")
        
        for check_func in self.checks:
            start_time = time.time()
            
            try:
                result = check_func()
                result.duration = time.time() - start_time
                results.append(result)
                
                logger.debug(f"Health check {result.name}: {result.status} ({result.duration:.3f}s)")
                
            except Exception as e:
                result = HealthCheckResult(
                    name=check_func.__name__.replace('_check_', ''),
                    status="error",
                    message=f"Health check failed: {e}",
                    duration=time.time() - start_time
                )
                results.append(result)
                logger.error(f"Health check {result.name} failed: {e}")
        
        logger.info(f"Health check completed: {self._summarize_results(results)}")
        return results
    
    def _summarize_results(self, results: List[HealthCheckResult]) -> str:
        """Summarize health check results."""
        ok_count = sum(1 for r in results if r.status == "ok")
        warning_count = sum(1 for r in results if r.status == "warning")
        error_count = sum(1 for r in results if r.status == "error")
        
        return f"{ok_count} OK, {warning_count} warnings, {error_count} errors"
    
    def _check_system_info(self) -> HealthCheckResult:
        """Check system information."""
        try:
            info = {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "memory_available": self._get_memory_info()
            }
            
            # Check Python version
            if sys.version_info < (3, 9):
                return HealthCheckResult(
                    name="system_info",
                    status="error",
                    message="Python 3.9+ required",
                    details=info
                )
            
            return HealthCheckResult(
                name="system_info",
                status="ok",
                message="System information collected",
                details=info
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_info",
                status="error",
                message=f"Failed to collect system info: {e}"
            )
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check required dependencies."""
        required_modules = [
            "numpy", "scipy", "networkx", "pydantic", "click", "rich", "typer"
        ]
        
        missing = []
        available = []
        
        for module in required_modules:
            try:
                __import__(module)
                available.append(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            return HealthCheckResult(
                name="dependencies",
                status="error",
                message=f"Missing required modules: {missing}",
                details={"available": available, "missing": missing}
            )
        
        return HealthCheckResult(
            name="dependencies",
            status="ok",
            message=f"All {len(available)} required modules available",
            details={"available": available}
        )
    
    def _check_scheduler_basic(self) -> HealthCheckResult:
        """Check basic scheduler functionality."""
        try:
            # Test scheduler initialization
            scheduler = QuantumScheduler(backend="classical")
            
            # Test empty scheduling
            solution = scheduler.schedule([], [])
            if solution.assignments != {}:
                return HealthCheckResult(
                    name="scheduler_basic",
                    status="error",
                    message="Empty scheduling failed"
                )
            
            # Test basic scheduling
            agents = [Agent(id="test_agent", skills=["test"], capacity=1)]
            tasks = [Task(id="test_task", required_skills=["test"], duration=1, priority=1)]
            
            solution = scheduler.schedule(agents, tasks, {"skill_match_required": True})
            
            if len(solution.assignments) != 1:
                return HealthCheckResult(
                    name="scheduler_basic",
                    status="error",
                    message="Basic scheduling failed"
                )
            
            return HealthCheckResult(
                name="scheduler_basic",
                status="ok",
                message="Basic scheduler functionality working",
                details={"test_assignments": len(solution.assignments)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="scheduler_basic",
                status="error",
                message=f"Scheduler test failed: {e}"
            )
    
    def _check_backend_availability(self) -> HealthCheckResult:
        """Check backend availability."""
        try:
            backend = ClassicalBackend()
            
            if not backend.is_available():
                return HealthCheckResult(
                    name="backend_availability",
                    status="error",
                    message="Classical backend not available"
                )
            
            capabilities = backend.get_capabilities()
            
            return HealthCheckResult(
                name="backend_availability",
                status="ok",
                message="Classical backend available",
                details=capabilities
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="backend_availability",
                status="error",
                message=f"Backend check failed: {e}"
            )
    
    def _check_metrics_system(self) -> HealthCheckResult:
        """Check metrics collection system."""
        try:
            collector = get_metrics_collector()
            
            # Test metrics collection
            collector.start_operation("health_check", 1, 1)
            collector.end_operation("health_check", True, "test", 1, 1.0, 1.0)
            
            metrics = collector.get_system_metrics()
            
            return HealthCheckResult(
                name="metrics_system",
                status="ok",
                message="Metrics system operational",
                details={
                    "total_operations": metrics.total_operations,
                    "success_rate": metrics.success_rate
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="metrics_system",
                status="error",  
                message=f"Metrics system failed: {e}"
            )
    
    def _check_validation_system(self) -> HealthCheckResult:
        """Check input validation system."""
        try:
            from .core.validators import InputValidator
            from .security import SecuritySanitizer
            
            # Test agent validation
            agent = Agent(id="test", skills=["python"], capacity=1)
            validated_agent = InputValidator.validate_agent(agent)
            
            # Test task validation
            task = Task(id="test", required_skills=["python"], duration=1, priority=1)
            validated_task = InputValidator.validate_task(task)
            
            # Test sanitizer
            safe_id = SecuritySanitizer.sanitize_id("test_id_123")
            
            return HealthCheckResult(
                name="validation_system",
                status="ok",
                message="Validation and security systems operational"
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="validation_system",
                status="error",
                message=f"Validation system failed: {e}"
            )
    
    def _check_performance(self) -> HealthCheckResult:
        """Check performance characteristics."""
        try:
            scheduler = QuantumScheduler(backend="classical")
            
            # Test small problem performance
            agents = [Agent(id=f"agent{i}", skills=["test"], capacity=2) for i in range(5)]
            tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1) for i in range(10)]
            
            start_time = time.time()
            solution = scheduler.schedule(agents, tasks)
            solve_time = time.time() - start_time
            
            # Performance thresholds
            if solve_time > 5.0:
                status = "warning"
                message = f"Slow performance: {solve_time:.3f}s for small problem"
            elif solve_time > 1.0:
                status = "warning"
                message = f"Acceptable performance: {solve_time:.3f}s"
            else:
                status = "ok"
                message = f"Good performance: {solve_time:.3f}s"
            
            return HealthCheckResult(
                name="performance",
                status=status,
                message=message,
                details={
                    "solve_time": solve_time,
                    "agents": len(agents),
                    "tasks": len(tasks),
                    "assignments": len(solution.assignments)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="performance",
                status="error",
                message=f"Performance test failed: {e}"
            )
    
    def _get_memory_info(self) -> str:
        """Get available memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.available / (1024**3):.1f} GB available"
        except ImportError:
            return "Memory info unavailable (psutil not installed)"
        except Exception:
            return "Memory info unavailable"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status information
        """
        results = self.run_all_checks()
        
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        overall_status = "ok"
        if status_counts.get("error", 0) > 0:
            overall_status = "error"
        elif status_counts.get("warning", 0) > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "checks_run": len(results),
            "status_counts": status_counts,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in results
            ]
        }