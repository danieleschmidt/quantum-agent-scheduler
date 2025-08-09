"""Multi-Region Quantum Cloud Orchestration for Massive-Scale Agent Scheduling.

This module implements a sophisticated quantum cloud orchestration system that
can dynamically distribute workloads across multiple quantum computing providers
and regions for maximum scalability and availability.
"""

import logging
import time
import asyncio
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported quantum cloud providers."""
    AWS_BRAKET = "aws_braket"
    IBM_QUANTUM = "ibm_quantum"
    GOOGLE_QUANTUM_AI = "google_quantum_ai"
    AZURE_QUANTUM = "azure_quantum"
    DWAVE = "dwave"
    IONQ = "ionq"
    RIGETTI = "rigetti"
    XANADU = "xanadu"


class Region(Enum):
    """Geographic regions for quantum cloud deployment."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"  
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    CA_CENTRAL_1 = "ca-central-1"


class WorkloadType(Enum):
    """Types of quantum workloads."""
    SMALL_QUBO = "small_qubo"      # < 50 qubits
    MEDIUM_QUBO = "medium_qubo"    # 50-200 qubits
    LARGE_QUBO = "large_qubo"      # 200-1000 qubits
    ULTRA_LARGE = "ultra_large"    # > 1000 qubits
    REAL_TIME = "real_time"        # Time-critical workloads
    BATCH = "batch"                # Batch processing


@dataclass
class CloudResource:
    """Definition of a quantum cloud resource."""
    provider: CloudProvider
    region: Region
    device_name: str
    max_qubits: int
    queue_length: int = 0
    availability: bool = True
    cost_per_shot: float = 0.0001
    latency_ms: float = 1000.0
    reliability_score: float = 0.95
    specialized_features: List[str] = field(default_factory=list)
    
    @property
    def utilization_score(self) -> float:
        """Calculate utilization score for load balancing."""
        queue_penalty = min(self.queue_length / 100.0, 1.0)  # Normalize queue length
        latency_penalty = min(self.latency_ms / 10000.0, 1.0)  # Normalize latency
        cost_penalty = min(self.cost_per_shot / 0.01, 1.0)  # Normalize cost
        
        return self.reliability_score * (1.0 - queue_penalty - latency_penalty - cost_penalty * 0.5)


@dataclass
class WorkloadRequest:
    """Request for quantum workload processing."""
    request_id: str
    workload_type: WorkloadType
    qubo_matrix: Optional[Any] = None
    priority: int = 5  # 1-10, higher = more priority
    max_execution_time: float = 300.0  # seconds
    max_cost: float = 10.0  # dollars
    preferred_providers: List[CloudProvider] = field(default_factory=list)
    preferred_regions: List[Region] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    @property
    def estimated_qubits(self) -> int:
        """Estimate number of qubits needed."""
        if self.qubo_matrix is not None and hasattr(self.qubo_matrix, 'shape'):
            return self.qubo_matrix.shape[0]
        
        # Estimate based on workload type
        if self.workload_type == WorkloadType.SMALL_QUBO:
            return 25
        elif self.workload_type == WorkloadType.MEDIUM_QUBO:
            return 100
        elif self.workload_type == WorkloadType.LARGE_QUBO:
            return 500
        elif self.workload_type == WorkloadType.ULTRA_LARGE:
            return 2000
        else:
            return 50


@dataclass
class ExecutionResult:
    """Result from quantum cloud execution."""
    request_id: str
    resource_used: CloudResource
    execution_time: float
    cost: float
    success: bool
    result_data: Dict[str, Any]
    error_message: str = ""
    queue_time: float = 0.0
    
    @property
    def total_time(self) -> float:
        """Total time including queue time."""
        return self.queue_time + self.execution_time


class LoadBalancer:
    """Intelligent load balancer for quantum cloud resources."""
    
    def __init__(self, balancing_strategy: str = "adaptive"):
        """Initialize load balancer.
        
        Args:
            balancing_strategy: Strategy for load balancing 
                               ("round_robin", "least_loaded", "cost_optimized", "adaptive")
        """
        self.balancing_strategy = balancing_strategy
        self.resource_history = defaultdict(deque)
        self.performance_metrics = defaultdict(dict)
        self.last_selection = defaultdict(int)
        
    def select_optimal_resource(self, 
                               request: WorkloadRequest,
                               available_resources: List[CloudResource]) -> Optional[CloudResource]:
        """Select optimal resource for workload execution.
        
        Args:
            request: Workload request
            available_resources: List of available resources
            
        Returns:
            Selected resource or None if no suitable resource found
        """
        if not available_resources:
            return None
        
        # Filter resources by requirements
        suitable_resources = self._filter_suitable_resources(request, available_resources)
        
        if not suitable_resources:
            logger.warning(f"No suitable resources found for request {request.request_id}")
            return None
        
        # Apply balancing strategy
        if self.balancing_strategy == "round_robin":
            selected = self._round_robin_selection(suitable_resources)
        elif self.balancing_strategy == "least_loaded":
            selected = self._least_loaded_selection(suitable_resources)
        elif self.balancing_strategy == "cost_optimized":
            selected = self._cost_optimized_selection(request, suitable_resources)
        else:  # adaptive
            selected = self._adaptive_selection(request, suitable_resources)
        
        # Update selection history
        self.last_selection[selected.provider.value] = time.time()
        
        logger.info(f"Selected resource: {selected.provider.value} in {selected.region.value} "
                   f"for request {request.request_id}")
        
        return selected
    
    def _filter_suitable_resources(self, 
                                  request: WorkloadRequest,
                                  resources: List[CloudResource]) -> List[CloudResource]:
        """Filter resources that meet request requirements."""
        suitable = []
        
        for resource in resources:
            # Check availability
            if not resource.availability:
                continue
            
            # Check qubit requirements
            if resource.max_qubits < request.estimated_qubits:
                continue
            
            # Check cost constraints
            estimated_cost = self._estimate_cost(request, resource)
            if estimated_cost > request.max_cost:
                continue
            
            # Check provider preferences
            if request.preferred_providers and resource.provider not in request.preferred_providers:
                continue
            
            # Check region preferences  
            if request.preferred_regions and resource.region not in request.preferred_regions:
                continue
            
            # Check specialized requirements
            if request.requirements.get('requires_annealing', False):
                if 'annealing' not in resource.specialized_features:
                    continue
            
            suitable.append(resource)
        
        return suitable
    
    def _estimate_cost(self, request: WorkloadRequest, resource: CloudResource) -> float:
        """Estimate execution cost for request on resource."""
        # Simplified cost estimation
        base_shots = request.requirements.get('num_shots', 1000)
        complexity_factor = request.estimated_qubits / 100.0
        
        estimated_cost = (base_shots * resource.cost_per_shot * complexity_factor)
        return estimated_cost
    
    def _round_robin_selection(self, resources: List[CloudResource]) -> CloudResource:
        """Round-robin resource selection."""
        # Simple round-robin based on provider
        providers = [r.provider.value for r in resources]
        last_times = {p: self.last_selection.get(p, 0) for p in providers}
        
        # Select provider used least recently
        selected_provider = min(last_times.items(), key=lambda x: x[1])[0]
        
        # Return first resource with that provider
        for resource in resources:
            if resource.provider.value == selected_provider:
                return resource
        
        return resources[0]  # Fallback
    
    def _least_loaded_selection(self, resources: List[CloudResource]) -> CloudResource:
        """Select least loaded resource."""
        return min(resources, key=lambda r: r.queue_length)
    
    def _cost_optimized_selection(self, 
                                 request: WorkloadRequest,
                                 resources: List[CloudResource]) -> CloudResource:
        """Select most cost-effective resource."""
        costs = [(r, self._estimate_cost(request, r)) for r in resources]
        return min(costs, key=lambda x: x[1])[0]
    
    def _adaptive_selection(self, 
                           request: WorkloadRequest,
                           resources: List[CloudResource]) -> CloudResource:
        """Adaptive resource selection based on multiple factors."""
        scores = []
        
        for resource in resources:
            # Base utilization score
            base_score = resource.utilization_score
            
            # Priority adjustment
            priority_boost = request.priority / 10.0 * 0.2
            
            # Time criticality adjustment
            if request.workload_type == WorkloadType.REAL_TIME:
                latency_penalty = resource.latency_ms / 10000.0
                base_score -= latency_penalty
            
            # Historical performance adjustment
            historical_success = self._get_historical_success_rate(resource)
            history_boost = (historical_success - 0.5) * 0.3  # Boost if > 50% success
            
            # Regional preference boost
            if resource.region in request.preferred_regions:
                base_score += 0.1
            
            total_score = base_score + priority_boost + history_boost
            scores.append((resource, total_score))
        
        # Select resource with highest score
        return max(scores, key=lambda x: x[1])[0]
    
    def _get_historical_success_rate(self, resource: CloudResource) -> float:
        """Get historical success rate for resource."""
        key = f"{resource.provider.value}_{resource.region.value}_{resource.device_name}"
        
        if key not in self.performance_metrics:
            return 0.8  # Default assumption
        
        metrics = self.performance_metrics[key]
        total_requests = metrics.get('total_requests', 1)
        successful_requests = metrics.get('successful_requests', 1)
        
        return successful_requests / total_requests
    
    def update_performance_metrics(self, 
                                  resource: CloudResource, 
                                  result: ExecutionResult) -> None:
        """Update performance metrics for resource."""
        key = f"{resource.provider.value}_{resource.region.value}_{resource.device_name}"
        
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_execution_time': 0.0,
                'total_cost': 0.0
            }
        
        metrics = self.performance_metrics[key]
        metrics['total_requests'] += 1
        
        if result.success:
            metrics['successful_requests'] += 1
        
        metrics['total_execution_time'] += result.execution_time
        metrics['total_cost'] += result.cost


class CloudOrchestrator:
    """Main orchestrator for multi-region quantum cloud operations."""
    
    def __init__(self, 
                 max_concurrent_jobs: int = 100,
                 resource_refresh_interval: float = 60.0,
                 enable_auto_scaling: bool = True):
        """Initialize cloud orchestrator.
        
        Args:
            max_concurrent_jobs: Maximum concurrent job executions
            resource_refresh_interval: Interval for resource status refresh (seconds)
            enable_auto_scaling: Enable automatic scaling of resources
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_refresh_interval = resource_refresh_interval
        self.enable_auto_scaling = enable_auto_scaling
        
        # Core components
        self.load_balancer = LoadBalancer(balancing_strategy="adaptive")
        self.available_resources = []
        self.job_queue = deque()
        self.active_jobs = {}
        self.completed_jobs = {}
        
        # Resource monitoring
        self.resource_monitors = {}
        self.last_resource_refresh = 0.0
        
        # Performance tracking
        self.orchestration_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_execution_time': 0.0,
            'total_queue_time': 0.0,
            'total_cost': 0.0
        }
        
        # Auto-scaling parameters
        self.scaling_thresholds = {
            'queue_length_threshold': 50,
            'avg_wait_time_threshold': 300.0,  # 5 minutes
            'resource_utilization_threshold': 0.8
        }
        
        # Initialize with default resources
        self._initialize_default_resources()
        
        logger.info(f"Initialized CloudOrchestrator with {len(self.available_resources)} resources")
    
    def _initialize_default_resources(self) -> None:
        """Initialize with default quantum cloud resources."""
        default_resources = [
            # AWS Braket resources
            CloudResource(CloudProvider.AWS_BRAKET, Region.US_EAST_1, "SV1", 34, 
                         cost_per_shot=0.00075, latency_ms=2000, reliability_score=0.95,
                         specialized_features=["gate_based"]),
            CloudResource(CloudProvider.AWS_BRAKET, Region.US_WEST_2, "Advantage_system6.1", 5000,
                         cost_per_shot=0.00019, latency_ms=5000, reliability_score=0.92,
                         specialized_features=["annealing", "large_scale"]),
            
            # IBM Quantum resources
            CloudResource(CloudProvider.IBM_QUANTUM, Region.US_EAST_1, "ibmq_qasm_simulator", 32,
                         cost_per_shot=0.0, latency_ms=1500, reliability_score=0.98,
                         specialized_features=["simulator", "gate_based"]),
            CloudResource(CloudProvider.IBM_QUANTUM, Region.EU_WEST_1, "ibm_lagos", 127,
                         cost_per_shot=0.002, latency_ms=8000, reliability_score=0.89,
                         specialized_features=["gate_based", "large_scale"]),
            
            # Google Quantum AI
            CloudResource(CloudProvider.GOOGLE_QUANTUM_AI, Region.US_WEST_2, "rainbow", 70,
                         cost_per_shot=0.001, latency_ms=3000, reliability_score=0.93,
                         specialized_features=["gate_based", "high_fidelity"]),
            
            # Azure Quantum
            CloudResource(CloudProvider.AZURE_QUANTUM, Region.EU_CENTRAL_1, "ionq.simulator", 29,
                         cost_per_shot=0.0001, latency_ms=1000, reliability_score=0.96,
                         specialized_features=["simulator", "ion_trap"]),
            
            # D-Wave (specialized for annealing)
            CloudResource(CloudProvider.DWAVE, Region.US_WEST_2, "Advantage_system4.1", 5640,
                         cost_per_shot=0.00005, latency_ms=4000, reliability_score=0.94,
                         specialized_features=["annealing", "ultra_large_scale"]),
        ]
        
        self.available_resources = default_resources
    
    async def submit_workload(self, request: WorkloadRequest) -> str:
        """Submit workload for execution.
        
        Args:
            request: Workload request
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        request.request_id = job_id
        
        # Add to job queue
        self.job_queue.append(request)
        self.orchestration_metrics['total_requests'] += 1
        
        logger.info(f"Submitted workload {job_id} ({request.workload_type.value}) to queue")
        
        # Trigger job processing
        asyncio.create_task(self._process_job_queue())
        
        return job_id
    
    async def _process_job_queue(self) -> None:
        """Process pending jobs in the queue."""
        while self.job_queue and len(self.active_jobs) < self.max_concurrent_jobs:
            if not self.available_resources:
                logger.warning("No available resources for job processing")
                break
            
            request = self.job_queue.popleft()
            
            # Select optimal resource
            selected_resource = self.load_balancer.select_optimal_resource(
                request, self.available_resources
            )
            
            if selected_resource is None:
                # Put back in queue and try later
                self.job_queue.appendleft(request)
                logger.warning(f"No suitable resource found for job {request.request_id}")
                await asyncio.sleep(5.0)  # Wait before retrying
                continue
            
            # Start job execution
            job_task = asyncio.create_task(
                self._execute_job(request, selected_resource)
            )
            
            self.active_jobs[request.request_id] = {
                'request': request,
                'resource': selected_resource,
                'task': job_task,
                'start_time': time.time()
            }
            
            logger.info(f"Started execution of job {request.request_id} "
                       f"on {selected_resource.provider.value}")
    
    async def _execute_job(self, 
                          request: WorkloadRequest,
                          resource: CloudResource) -> ExecutionResult:
        """Execute job on selected resource.
        
        Args:
            request: Workload request
            resource: Selected resource
            
        Returns:
            Execution result
        """
        job_start_time = time.time()
        queue_time = job_start_time - request.created_at
        
        try:
            # Simulate quantum execution (replace with actual quantum backend calls)
            execution_result = await self._simulate_quantum_execution(request, resource)
            
            execution_time = time.time() - job_start_time
            
            result = ExecutionResult(
                request_id=request.request_id,
                resource_used=resource,
                execution_time=execution_time,
                cost=execution_result['cost'],
                success=True,
                result_data=execution_result['data'],
                queue_time=queue_time
            )
            
            # Update metrics
            self.orchestration_metrics['successful_requests'] += 1
            self.orchestration_metrics['total_execution_time'] += execution_time
            self.orchestration_metrics['total_queue_time'] += queue_time
            self.orchestration_metrics['total_cost'] += result.cost
            
            self.load_balancer.update_performance_metrics(resource, result)
            
        except Exception as e:
            execution_time = time.time() - job_start_time
            
            result = ExecutionResult(
                request_id=request.request_id,
                resource_used=resource,
                execution_time=execution_time,
                cost=0.0,
                success=False,
                result_data={},
                error_message=str(e),
                queue_time=queue_time
            )
            
            logger.error(f"Job {request.request_id} failed: {e}")
        
        # Move from active to completed
        if request.request_id in self.active_jobs:
            del self.active_jobs[request.request_id]
        
        self.completed_jobs[request.request_id] = result
        
        # Trigger auto-scaling check
        if self.enable_auto_scaling:
            await self._check_auto_scaling()
        
        return result
    
    async def _simulate_quantum_execution(self, 
                                        request: WorkloadRequest,
                                        resource: CloudResource) -> Dict[str, Any]:
        """Simulate quantum execution (replace with real implementations)."""
        # Simulate execution delay based on workload size and resource
        base_time = request.estimated_qubits * 0.01
        resource_factor = 1.0 / resource.reliability_score
        execution_delay = base_time * resource_factor
        
        await asyncio.sleep(min(execution_delay, 2.0))  # Cap simulation time
        
        # Simulate cost calculation
        num_shots = request.requirements.get('num_shots', 1000)
        cost = num_shots * resource.cost_per_shot
        
        # Simulate result data
        result_data = {
            'solution_vector': [0, 1] * (request.estimated_qubits // 2),
            'energy': -5.5 + request.estimated_qubits * 0.1,
            'success_probability': resource.reliability_score
        }
        
        # Simulate occasional failures
        if resource.reliability_score < 0.9 and time.time() % 10 < 1:
            raise RuntimeError("Simulated quantum execution failure")
        
        return {
            'cost': cost,
            'data': result_data
        }
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a submitted job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        # Check if job is active
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                'job_id': job_id,
                'status': 'running',
                'resource': f"{job_info['resource'].provider.value}",
                'start_time': job_info['start_time'],
                'elapsed_time': time.time() - job_info['start_time']
            }
        
        # Check if job is completed
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                'job_id': job_id,
                'status': 'completed' if result.success else 'failed',
                'success': result.success,
                'execution_time': result.execution_time,
                'queue_time': result.queue_time,
                'cost': result.cost,
                'error_message': result.error_message
            }
        
        # Check if job is in queue
        for request in self.job_queue:
            if request.request_id == job_id:
                return {
                    'job_id': job_id,
                    'status': 'queued',
                    'queue_position': list(self.job_queue).index(request) + 1,
                    'estimated_wait_time': self._estimate_queue_wait_time()
                }
        
        return None  # Job not found
    
    def _estimate_queue_wait_time(self) -> float:
        """Estimate wait time for jobs in queue."""
        if not self.job_queue:
            return 0.0
        
        # Simple estimation based on current active jobs
        avg_execution_time = 30.0  # Default estimate
        
        if self.orchestration_metrics['successful_requests'] > 0:
            avg_execution_time = (
                self.orchestration_metrics['total_execution_time'] /
                self.orchestration_metrics['successful_requests']
            )
        
        available_slots = self.max_concurrent_jobs - len(self.active_jobs)
        
        if available_slots > 0:
            return len(self.job_queue) * avg_execution_time / available_slots
        else:
            return len(self.job_queue) * avg_execution_time
    
    async def _check_auto_scaling(self) -> None:
        """Check if auto-scaling is needed."""
        if not self.enable_auto_scaling:
            return
        
        current_time = time.time()
        
        # Check queue length threshold
        if len(self.job_queue) > self.scaling_thresholds['queue_length_threshold']:
            logger.info(f"Queue length ({len(self.job_queue)}) exceeded threshold, "
                       "considering resource scaling")
            await self._scale_resources(scale_up=True)
        
        # Check average wait time
        avg_wait_time = self._estimate_queue_wait_time()
        if avg_wait_time > self.scaling_thresholds['avg_wait_time_threshold']:
            logger.info(f"Average wait time ({avg_wait_time:.1f}s) exceeded threshold, "
                       "considering resource scaling")
            await self._scale_resources(scale_up=True)
        
        # Check resource utilization
        utilization = len(self.active_jobs) / max(self.max_concurrent_jobs, 1)
        if utilization > self.scaling_thresholds['resource_utilization_threshold']:
            logger.info(f"Resource utilization ({utilization:.1%}) exceeded threshold, "
                       "considering capacity scaling")
            await self._scale_capacity(scale_up=True)
    
    async def _scale_resources(self, scale_up: bool = True) -> None:
        """Scale quantum resources up or down."""
        if scale_up:
            # Try to discover new resources from providers
            new_resources = await self._discover_new_resources()
            if new_resources:
                self.available_resources.extend(new_resources)
                logger.info(f"Added {len(new_resources)} new quantum resources")
        else:
            # Remove underperforming resources
            self._remove_underperforming_resources()
    
    async def _scale_capacity(self, scale_up: bool = True) -> None:
        """Scale execution capacity up or down."""
        if scale_up:
            # Increase concurrent job limit
            old_limit = self.max_concurrent_jobs
            self.max_concurrent_jobs = min(self.max_concurrent_jobs * 2, 500)
            logger.info(f"Scaled concurrent job limit from {old_limit} to {self.max_concurrent_jobs}")
            
            # Process any queued jobs
            await self._process_job_queue()
        else:
            # Decrease concurrent job limit
            self.max_concurrent_jobs = max(self.max_concurrent_jobs // 2, 10)
    
    async def _discover_new_resources(self) -> List[CloudResource]:
        """Discover new quantum resources from providers."""
        # Simulate resource discovery
        new_resources = []
        
        # Add simulators as fallback resources
        simulators = [
            CloudResource(CloudProvider.IBM_QUANTUM, Region.US_EAST_1, "qasm_simulator_backup", 32,
                         cost_per_shot=0.0, latency_ms=800, reliability_score=0.99,
                         specialized_features=["simulator"]),
            CloudResource(CloudProvider.AWS_BRAKET, Region.EU_WEST_1, "SV1_backup", 34,
                         cost_per_shot=0.00075, latency_ms=1800, reliability_score=0.96,
                         specialized_features=["gate_based", "backup"])
        ]
        
        # Only add resources that aren't already available
        existing_devices = {(r.provider, r.device_name) for r in self.available_resources}
        
        for resource in simulators:
            if (resource.provider, resource.device_name) not in existing_devices:
                new_resources.append(resource)
        
        return new_resources
    
    def _remove_underperforming_resources(self) -> None:
        """Remove resources with poor performance."""
        # Identify underperforming resources
        to_remove = []
        
        for resource in self.available_resources:
            success_rate = self.load_balancer._get_historical_success_rate(resource)
            if success_rate < 0.5:  # Less than 50% success rate
                to_remove.append(resource)
        
        # Remove underperforming resources
        for resource in to_remove:
            self.available_resources.remove(resource)
            logger.info(f"Removed underperforming resource: {resource.device_name}")
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics."""
        current_time = time.time()
        
        # Calculate success rate
        success_rate = (
            self.orchestration_metrics['successful_requests'] / 
            max(self.orchestration_metrics['total_requests'], 1)
        )
        
        # Calculate average times
        avg_execution_time = (
            self.orchestration_metrics['total_execution_time'] /
            max(self.orchestration_metrics['successful_requests'], 1)
        )
        
        avg_queue_time = (
            self.orchestration_metrics['total_queue_time'] /
            max(self.orchestration_metrics['total_requests'], 1)
        )
        
        return {
            'total_requests': self.orchestration_metrics['total_requests'],
            'successful_requests': self.orchestration_metrics['successful_requests'],
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'average_queue_time': avg_queue_time,
            'total_cost': self.orchestration_metrics['total_cost'],
            'active_jobs': len(self.active_jobs),
            'queued_jobs': len(self.job_queue),
            'available_resources': len(self.available_resources),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'current_utilization': len(self.active_jobs) / max(self.max_concurrent_jobs, 1)
        }
    
    def get_resource_status(self) -> List[Dict[str, Any]]:
        """Get status of all available resources."""
        return [
            {
                'provider': resource.provider.value,
                'region': resource.region.value,
                'device_name': resource.device_name,
                'max_qubits': resource.max_qubits,
                'queue_length': resource.queue_length,
                'availability': resource.availability,
                'cost_per_shot': resource.cost_per_shot,
                'latency_ms': resource.latency_ms,
                'reliability_score': resource.reliability_score,
                'utilization_score': resource.utilization_score,
                'specialized_features': resource.specialized_features,
                'historical_success_rate': self.load_balancer._get_historical_success_rate(resource)
            }
            for resource in self.available_resources
        ]
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        logger.info("Shutting down cloud orchestrator...")
        
        # Wait for active jobs to complete (with timeout)
        shutdown_timeout = 300.0  # 5 minutes
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete...")
            await asyncio.sleep(5.0)
        
        # Cancel remaining jobs if timeout reached
        if self.active_jobs:
            logger.warning(f"Forcefully cancelling {len(self.active_jobs)} remaining jobs")
            for job_id, job_info in self.active_jobs.items():
                job_info['task'].cancel()
        
        logger.info("Cloud orchestrator shutdown complete")


class MultiRegionScheduler:
    """High-level multi-region quantum scheduler interface."""
    
    def __init__(self, 
                 enable_multi_region: bool = True,
                 preferred_regions: Optional[List[Region]] = None):
        """Initialize multi-region scheduler.
        
        Args:
            enable_multi_region: Enable multi-region deployment
            preferred_regions: Preferred regions for deployment
        """
        self.enable_multi_region = enable_multi_region
        self.preferred_regions = preferred_regions or [Region.US_EAST_1, Region.EU_WEST_1]
        
        # Initialize orchestrator
        self.orchestrator = CloudOrchestrator(
            max_concurrent_jobs=200,
            enable_auto_scaling=True
        )
        
        logger.info(f"Initialized MultiRegionScheduler with regions: {[r.value for r in self.preferred_regions]}")
    
    async def schedule_agents_at_scale(self, 
                                     agents: List,
                                     tasks: List,
                                     constraints: Optional[Dict[str, Any]] = None,
                                     priority: int = 5) -> str:
        """Schedule agents at massive scale across quantum cloud.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            constraints: Scheduling constraints
            priority: Job priority (1-10)
            
        Returns:
            Job ID for tracking
        """
        # Determine workload characteristics
        problem_size = len(agents) + len(tasks)
        
        if problem_size < 50:
            workload_type = WorkloadType.SMALL_QUBO
        elif problem_size < 200:
            workload_type = WorkloadType.MEDIUM_QUBO
        elif problem_size < 1000:
            workload_type = WorkloadType.LARGE_QUBO
        else:
            workload_type = WorkloadType.ULTRA_LARGE
        
        # Create workload request
        request = WorkloadRequest(
            request_id="",  # Will be assigned by orchestrator
            workload_type=workload_type,
            priority=priority,
            max_execution_time=600.0,  # 10 minutes
            max_cost=50.0,  # $50 max
            preferred_regions=self.preferred_regions,
            requirements={
                'num_shots': min(max(problem_size * 10, 1000), 10000),
                'optimization_level': 3 if problem_size > 100 else 2,
                'agents': len(agents),
                'tasks': len(tasks),
                'constraints': constraints or {}
            }
        )
        
        # Submit to orchestrator
        job_id = await self.orchestrator.submit_workload(request)
        
        logger.info(f"Submitted large-scale scheduling job {job_id} "
                   f"({len(agents)} agents, {len(tasks)} tasks)")
        
        return job_id
    
    async def get_result(self, job_id: str, timeout: float = 600.0) -> Optional[Dict[str, Any]]:
        """Get result from completed job.
        
        Args:
            job_id: Job identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Job result or None if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            status = await self.orchestrator.get_job_status(job_id)
            
            if status is None:
                return None  # Job not found
            
            if status['status'] == 'completed':
                result = self.orchestrator.completed_jobs.get(job_id)
                if result:
                    return {
                        'success': result.success,
                        'execution_time': result.execution_time,
                        'queue_time': result.queue_time,
                        'cost': result.cost,
                        'provider': result.resource_used.provider.value,
                        'region': result.resource_used.region.value,
                        'result_data': result.result_data
                    }
            
            elif status['status'] == 'failed':
                result = self.orchestrator.completed_jobs.get(job_id)
                return {
                    'success': False,
                    'error': result.error_message if result else 'Unknown error'
                }
            
            # Wait before checking again
            await asyncio.sleep(5.0)
        
        return None  # Timeout
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global orchestration metrics."""
        orchestration_metrics = self.orchestrator.get_orchestration_metrics()
        resource_status = self.orchestrator.get_resource_status()
        
        # Aggregate by provider and region
        provider_stats = defaultdict(lambda: {'requests': 0, 'success_rate': 0})
        region_stats = defaultdict(lambda: {'resources': 0, 'total_qubits': 0})
        
        for resource in resource_status:
            provider = resource['provider']
            region = resource['region']
            
            region_stats[region]['resources'] += 1
            region_stats[region]['total_qubits'] += resource['max_qubits']
            
            provider_stats[provider]['success_rate'] = resource['historical_success_rate']
        
        return {
            'orchestration': orchestration_metrics,
            'provider_statistics': dict(provider_stats),
            'region_statistics': dict(region_stats),
            'total_quantum_resources': len(resource_status),
            'total_available_qubits': sum(r['max_qubits'] for r in resource_status),
            'multi_region_enabled': self.enable_multi_region,
            'preferred_regions': [r.value for r in self.preferred_regions]
        }
    
    async def shutdown(self) -> None:
        """Shutdown multi-region scheduler."""
        await self.orchestrator.shutdown()