"""Multi-region deployment and data residency support."""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported cloud regions."""
    US_EAST_1 = "us-east-1"        # US East (N. Virginia)
    US_WEST_2 = "us-west-2"        # US West (Oregon)
    EU_WEST_1 = "eu-west-1"        # Europe (Ireland)
    EU_CENTRAL_1 = "eu-central-1"  # Europe (Frankfurt)
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Asia Pacific (Singapore)
    AP_NORTHEAST_1 = "ap-northeast-1"  # Asia Pacific (Tokyo)
    CA_CENTRAL_1 = "ca-central-1"  # Canada (Central)
    AU_SOUTHEAST_2 = "au-southeast-2"  # Australia (Sydney)


class DataResidencyRule(Enum):
    """Data residency requirements."""
    GDPR_EU_ONLY = "gdpr_eu_only"           # EU data must stay in EU
    CCPA_US_PREFERENCE = "ccpa_us_preference"  # California data preferably in US
    CHINA_LOCAL_ONLY = "china_local_only"   # China data must stay in China
    RUSSIA_LOCAL_ONLY = "russia_local_only" # Russia data must stay in Russia
    HEALTHCARE_STRICT = "healthcare_strict" # Healthcare data strict residency
    FINANCIAL_STRICT = "financial_strict"   # Financial data strict residency


@dataclass
class RegionInfo:
    """Information about a cloud region."""
    region: Region
    name: str
    country_code: str
    continent: str
    data_center_locations: List[str]
    compliance_certifications: List[str]
    latency_zones: List[str]
    available_services: List[str]
    cost_multiplier: float = 1.0
    quantum_backends_available: bool = False


@dataclass
class DataResidencyPolicy:
    """Data residency policy configuration."""
    policy_id: str
    name: str
    applicable_regions: List[Region]
    allowed_target_regions: List[Region]
    data_categories: List[str]
    compliance_requirements: List[str]
    max_latency_ms: Optional[int] = None
    backup_allowed_regions: List[Region] = None
    encryption_required: bool = True
    
    def __post_init__(self):
        if self.backup_allowed_regions is None:
            self.backup_allowed_regions = []


class MultiRegionManager:
    """Manages multi-region deployments and data residency."""
    
    def __init__(self):
        """Initialize multi-region manager."""
        self._regions = self._initialize_regions()
        self._residency_policies: List[DataResidencyPolicy] = []
        self._region_preferences: Dict[str, List[Region]] = {}
        self._latency_matrix: Dict[Tuple[Region, Region], float] = {}
        self._region_health: Dict[Region, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Initialize default policies
        self._initialize_default_policies()
        
        # Initialize latency matrix with estimated values
        self._initialize_latency_matrix()
        
        logger.info(f"Multi-region manager initialized with {len(self._regions)} regions")
    
    def _initialize_regions(self) -> Dict[Region, RegionInfo]:
        """Initialize region information."""
        regions = {
            Region.US_EAST_1: RegionInfo(
                region=Region.US_EAST_1,
                name="US East (N. Virginia)",
                country_code="US",
                continent="North America",
                data_center_locations=["Ashburn", "Sterling"],
                compliance_certifications=["SOC", "PCI-DSS", "HIPAA", "FedRAMP"],
                latency_zones=["us-east", "americas"],
                available_services=["quantum_simulator", "classical_compute", "ml_services"],
                cost_multiplier=1.0,
                quantum_backends_available=True
            ),
            Region.US_WEST_2: RegionInfo(
                region=Region.US_WEST_2,
                name="US West (Oregon)",
                country_code="US",
                continent="North America",
                data_center_locations=["Boardman", "Umatilla"],
                compliance_certifications=["SOC", "PCI-DSS", "HIPAA"],
                latency_zones=["us-west", "americas"],
                available_services=["quantum_simulator", "classical_compute"],
                cost_multiplier=0.95,
                quantum_backends_available=True
            ),
            Region.EU_WEST_1: RegionInfo(
                region=Region.EU_WEST_1,
                name="Europe (Ireland)",
                country_code="IE",
                continent="Europe",
                data_center_locations=["Dublin"],
                compliance_certifications=["SOC", "PCI-DSS", "GDPR", "ISO27001"],
                latency_zones=["eu-west", "europe"],
                available_services=["quantum_simulator", "classical_compute"],
                cost_multiplier=1.1,
                quantum_backends_available=True
            ),
            Region.EU_CENTRAL_1: RegionInfo(
                region=Region.EU_CENTRAL_1,
                name="Europe (Frankfurt)",
                country_code="DE",
                continent="Europe",
                data_center_locations=["Frankfurt"],
                compliance_certifications=["SOC", "PCI-DSS", "GDPR", "ISO27001", "C5"],
                latency_zones=["eu-central", "europe"],
                available_services=["quantum_hardware", "quantum_simulator", "classical_compute"],
                cost_multiplier=1.15,
                quantum_backends_available=True
            ),
            Region.AP_SOUTHEAST_1: RegionInfo(
                region=Region.AP_SOUTHEAST_1,
                name="Asia Pacific (Singapore)",
                country_code="SG",
                continent="Asia",
                data_center_locations=["Singapore"],
                compliance_certifications=["SOC", "PCI-DSS", "MTCS", "PDPA"],
                latency_zones=["ap-southeast", "asia-pacific"],
                available_services=["quantum_simulator", "classical_compute"],
                cost_multiplier=1.05,
                quantum_backends_available=False
            ),
            Region.AP_NORTHEAST_1: RegionInfo(
                region=Region.AP_NORTHEAST_1,
                name="Asia Pacific (Tokyo)",
                country_code="JP",
                continent="Asia",
                data_center_locations=["Tokyo"],
                compliance_certifications=["SOC", "PCI-DSS", "ISMS"],
                latency_zones=["ap-northeast", "asia-pacific"],
                available_services=["quantum_simulator", "classical_compute"],
                cost_multiplier=1.2,
                quantum_backends_available=True
            ),
            Region.CA_CENTRAL_1: RegionInfo(
                region=Region.CA_CENTRAL_1,
                name="Canada (Central)",
                country_code="CA",
                continent="North America",
                data_center_locations=["Montreal"],
                compliance_certifications=["SOC", "PCI-DSS", "PIPEDA"],
                latency_zones=["ca-central", "americas"],
                available_services=["classical_compute"],
                cost_multiplier=1.05,
                quantum_backends_available=False
            ),
            Region.AU_SOUTHEAST_2: RegionInfo(
                region=Region.AU_SOUTHEAST_2,
                name="Australia (Sydney)",
                country_code="AU",
                continent="Oceania",
                data_center_locations=["Sydney"],
                compliance_certifications=["SOC", "PCI-DSS", "IRAP"],
                latency_zones=["au-southeast", "oceania"],
                available_services=["classical_compute"],
                cost_multiplier=1.25,
                quantum_backends_available=False
            )
        }
        
        return regions
    
    def _initialize_default_policies(self):
        """Initialize default data residency policies."""
        # GDPR policy - EU data stays in EU
        gdpr_policy = DataResidencyPolicy(
            policy_id="gdpr_eu_data",
            name="GDPR EU Data Residency",
            applicable_regions=[Region.EU_WEST_1, Region.EU_CENTRAL_1],
            allowed_target_regions=[Region.EU_WEST_1, Region.EU_CENTRAL_1],
            data_categories=["personal_data", "sensitive_data"],
            compliance_requirements=["GDPR"],
            max_latency_ms=50,
            encryption_required=True
        )
        
        # US data preference for CCPA
        ccpa_policy = DataResidencyPolicy(
            policy_id="ccpa_us_preference",
            name="CCPA US Data Preference",
            applicable_regions=[Region.US_EAST_1, Region.US_WEST_2, Region.CA_CENTRAL_1],
            allowed_target_regions=[Region.US_EAST_1, Region.US_WEST_2, Region.CA_CENTRAL_1],
            data_categories=["personal_data", "consumer_data"],
            compliance_requirements=["CCPA"],
            max_latency_ms=100,
            backup_allowed_regions=[Region.EU_WEST_1],  # For disaster recovery
            encryption_required=True
        )
        
        # Asia-Pacific data residency
        apac_policy = DataResidencyPolicy(
            policy_id="apac_data_residency",
            name="Asia-Pacific Data Residency",
            applicable_regions=[Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1, Region.AU_SOUTHEAST_2],
            allowed_target_regions=[Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1, Region.AU_SOUTHEAST_2],
            data_categories=["personal_data", "business_data"],
            compliance_requirements=["PDPA", "ISMS"],
            max_latency_ms=75,
            encryption_required=True
        )
        
        self._residency_policies.extend([gdpr_policy, ccpa_policy, apac_policy])
    
    def _initialize_latency_matrix(self):
        """Initialize estimated latency matrix between regions."""
        # Estimated latencies in milliseconds (one-way)
        latencies = {
            (Region.US_EAST_1, Region.US_WEST_2): 65,
            (Region.US_EAST_1, Region.EU_WEST_1): 75,
            (Region.US_EAST_1, Region.EU_CENTRAL_1): 85,
            (Region.US_EAST_1, Region.AP_SOUTHEAST_1): 180,
            (Region.US_EAST_1, Region.AP_NORTHEAST_1): 140,
            (Region.US_EAST_1, Region.CA_CENTRAL_1): 35,
            (Region.US_EAST_1, Region.AU_SOUTHEAST_2): 200,
            
            (Region.US_WEST_2, Region.EU_WEST_1): 140,
            (Region.US_WEST_2, Region.EU_CENTRAL_1): 150,
            (Region.US_WEST_2, Region.AP_SOUTHEAST_1): 160,
            (Region.US_WEST_2, Region.AP_NORTHEAST_1): 100,
            (Region.US_WEST_2, Region.CA_CENTRAL_1): 45,
            (Region.US_WEST_2, Region.AU_SOUTHEAST_2): 140,
            
            (Region.EU_WEST_1, Region.EU_CENTRAL_1): 25,
            (Region.EU_WEST_1, Region.AP_SOUTHEAST_1): 170,
            (Region.EU_WEST_1, Region.AP_NORTHEAST_1): 220,
            (Region.EU_WEST_1, Region.CA_CENTRAL_1): 85,
            (Region.EU_WEST_1, Region.AU_SOUTHEAST_2): 280,
            
            (Region.EU_CENTRAL_1, Region.AP_SOUTHEAST_1): 160,
            (Region.EU_CENTRAL_1, Region.AP_NORTHEAST_1): 200,
            (Region.EU_CENTRAL_1, Region.CA_CENTRAL_1): 95,
            (Region.EU_CENTRAL_1, Region.AU_SOUTHEAST_2): 270,
            
            (Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1): 70,
            (Region.AP_SOUTHEAST_1, Region.CA_CENTRAL_1): 190,
            (Region.AP_SOUTHEAST_1, Region.AU_SOUTHEAST_2): 90,
            
            (Region.AP_NORTHEAST_1, Region.CA_CENTRAL_1): 130,
            (Region.AP_NORTHEAST_1, Region.AU_SOUTHEAST_2): 120,
            
            (Region.CA_CENTRAL_1, Region.AU_SOUTHEAST_2): 210
        }
        
        # Add reverse mappings and self-latencies
        for (source, target), latency in list(latencies.items()):
            self._latency_matrix[(source, target)] = latency
            self._latency_matrix[(target, source)] = latency
        
        # Self-latencies
        for region in Region:
            self._latency_matrix[(region, region)] = 1.0
    
    def get_optimal_regions(
        self,
        user_location: Optional[str] = None,
        data_categories: Optional[List[str]] = None,
        compliance_requirements: Optional[List[str]] = None,
        quantum_required: bool = False,
        max_latency_ms: Optional[int] = None,
        cost_optimization: bool = False
    ) -> List[Region]:
        """Get optimal regions based on requirements.
        
        Args:
            user_location: User location (country code or region)
            data_categories: Categories of data to be processed
            compliance_requirements: Required compliance certifications
            quantum_required: Whether quantum backends are required
            max_latency_ms: Maximum acceptable latency
            cost_optimization: Whether to optimize for cost
            
        Returns:
            List of optimal regions ranked by suitability
        """
        with self._lock:
            candidates = list(self._regions.values())
            scored_regions = []
            
            for region_info in candidates:
                score = 0.0
                
                # Check quantum requirement
                if quantum_required and not region_info.quantum_backends_available:
                    continue
                
                # Check compliance requirements
                if compliance_requirements:
                    compliance_match = len(set(compliance_requirements) & set(region_info.compliance_certifications))
                    if compliance_match == 0:
                        continue
                    score += compliance_match * 10
                
                # Check data residency policies
                if data_categories:
                    residency_score = self._check_residency_compliance(region_info.region, data_categories)
                    if residency_score < 0:
                        continue  # Violates residency policy
                    score += residency_score
                
                # Latency scoring (if user location is provided)
                if user_location:
                    latency_score = self._calculate_latency_score(region_info, user_location, max_latency_ms)
                    if latency_score < 0:
                        continue  # Exceeds latency limit
                    score += latency_score
                
                # Cost scoring
                if cost_optimization:
                    cost_score = (2.0 - region_info.cost_multiplier) * 5  # Lower cost = higher score
                    score += cost_score
                
                # Service availability
                service_score = len(region_info.available_services)
                score += service_score
                
                # Health check (if available)
                health_score = self._get_region_health_score(region_info.region)
                score += health_score
                
                scored_regions.append((region_info.region, score))
            
            # Sort by score (descending)
            scored_regions.sort(key=lambda x: x[1], reverse=True)
            
            return [region for region, score in scored_regions]
    
    def _check_residency_compliance(self, region: Region, data_categories: List[str]) -> float:
        """Check if region complies with data residency policies."""
        score = 0.0
        
        for policy in self._residency_policies:
            # Check if any data categories match policy
            category_match = any(cat in policy.data_categories for cat in data_categories)
            if not category_match:
                continue
            
            # Check if region is in applicable regions
            if region in policy.applicable_regions:
                # This region originates data subject to policy
                if region in policy.allowed_target_regions:
                    score += 5.0  # Perfect match
                else:
                    return -1.0  # Policy violation
            elif region in policy.allowed_target_regions:
                score += 3.0  # Allowed target region
        
        return score
    
    def _calculate_latency_score(
        self,
        region_info: RegionInfo,
        user_location: str,
        max_latency_ms: Optional[int]
    ) -> float:
        """Calculate latency score for a region."""
        # Simple heuristic based on continent/country matching
        location_mapping = {
            "US": ["us-east", "us-west", "americas"],
            "CA": ["ca-central", "americas"],
            "GB": ["eu-west", "europe"],
            "DE": ["eu-central", "europe"],
            "FR": ["eu-west", "europe"],
            "SG": ["ap-southeast", "asia-pacific"],
            "JP": ["ap-northeast", "asia-pacific"],
            "AU": ["au-southeast", "oceania"]
        }
        
        user_zones = location_mapping.get(user_location.upper(), [])
        region_zones = region_info.latency_zones
        
        # Calculate zone overlap
        zone_overlap = len(set(user_zones) & set(region_zones))
        if zone_overlap > 0:
            base_score = zone_overlap * 10.0
        else:
            base_score = 1.0  # Minimal score for no overlap
        
        # Estimate latency and check against limit
        if max_latency_ms:
            # Simple estimation - could be replaced with actual measurements
            estimated_latency = self._estimate_latency(user_location, region_info.region)
            if estimated_latency > max_latency_ms:
                return -1.0  # Exceeds latency limit
            
            # Bonus for low latency
            latency_bonus = max(0, (max_latency_ms - estimated_latency) / max_latency_ms) * 5.0
            base_score += latency_bonus
        
        return base_score
    
    def _estimate_latency(self, user_location: str, region: Region) -> float:
        """Estimate latency from user location to region."""
        # Simplified estimation - in practice, use real measurements
        region_mapping = {
            "US": Region.US_EAST_1,
            "CA": Region.CA_CENTRAL_1,
            "GB": Region.EU_WEST_1,
            "DE": Region.EU_CENTRAL_1,
            "SG": Region.AP_SOUTHEAST_1,
            "JP": Region.AP_NORTHEAST_1,
            "AU": Region.AU_SOUTHEAST_2
        }
        
        source_region = region_mapping.get(user_location.upper(), Region.US_EAST_1)
        return self._latency_matrix.get((source_region, region), 200.0)
    
    def _get_region_health_score(self, region: Region) -> float:
        """Get health score for a region."""
        health_info = self._region_health.get(region, {})
        
        if not health_info:
            return 0.0  # No health info available
        
        # Calculate score based on health metrics
        uptime = health_info.get('uptime', 0.99)
        response_time = health_info.get('avg_response_time', 100)
        
        uptime_score = uptime * 5.0  # 0.99 uptime = 4.95 points
        response_score = max(0, (200 - response_time) / 200) * 3.0  # Lower response = higher score
        
        return uptime_score + response_score
    
    def validate_data_placement(
        self,
        source_region: Region,
        target_region: Region,
        data_categories: List[str]
    ) -> Dict[str, Any]:
        """Validate if data can be placed/moved between regions.
        
        Args:
            source_region: Source region where data originates
            target_region: Target region where data will be stored/processed
            data_categories: Categories of data being moved
            
        Returns:
            Validation result with compliance status
        """
        with self._lock:
            result = {
                "allowed": True,
                "violations": [],
                "warnings": [],
                "requirements": []
            }
            
            # Check each residency policy
            for policy in self._residency_policies:
                # Check if policy applies to this data
                category_match = any(cat in policy.data_categories for cat in data_categories)
                if not category_match:
                    continue
                
                # Check if source region is subject to this policy
                if source_region in policy.applicable_regions:
                    # Check if target region is allowed
                    if target_region not in policy.allowed_target_regions:
                        # Check backup regions
                        if target_region in policy.backup_allowed_regions:
                            result["warnings"].append(
                                f"Target region {target_region.value} is only allowed as backup "
                                f"under policy {policy.name}"
                            )
                        else:
                            result["allowed"] = False
                            result["violations"].append(
                                f"Policy {policy.name} prohibits moving data from "
                                f"{source_region.value} to {target_region.value}"
                            )
                    
                    # Check encryption requirements
                    if policy.encryption_required:
                        result["requirements"].append("Data must be encrypted in transit and at rest")
                    
                    # Check latency requirements
                    if policy.max_latency_ms:
                        estimated_latency = self._latency_matrix.get((source_region, target_region), 0)
                        if estimated_latency > policy.max_latency_ms:
                            result["warnings"].append(
                                f"Latency ({estimated_latency}ms) may exceed policy requirement "
                                f"({policy.max_latency_ms}ms)"
                            )
            
            return result
    
    def get_region_info(self, region: Region) -> Optional[RegionInfo]:
        """Get information about a specific region."""
        return self._regions.get(region)
    
    def add_residency_policy(self, policy: DataResidencyPolicy):
        """Add a custom data residency policy."""
        with self._lock:
            self._residency_policies.append(policy)
            logger.info(f"Added residency policy: {policy.name}")
    
    def update_region_health(self, region: Region, health_metrics: Dict[str, Any]):
        """Update health metrics for a region."""
        with self._lock:
            self._region_health[region] = {
                **health_metrics,
                "last_updated": time.time()
            }
    
    def get_cross_region_latency(self, source: Region, target: Region) -> float:
        """Get latency between two regions."""
        return self._latency_matrix.get((source, target), 0.0)
    
    def suggest_disaster_recovery_regions(
        self,
        primary_region: Region,
        max_regions: int = 3
    ) -> List[Region]:
        """Suggest disaster recovery regions for a primary region.
        
        Args:
            primary_region: Primary region to create DR for
            max_regions: Maximum number of DR regions to suggest
            
        Returns:
            List of suggested DR regions
        """
        with self._lock:
            primary_info = self._regions[primary_region]
            candidates = []
            
            for region, region_info in self._regions.items():
                if region == primary_region:
                    continue
                
                # Calculate suitability score
                score = 0.0
                
                # Prefer different continents for true DR
                if region_info.continent != primary_info.continent:
                    score += 10.0
                else:
                    score += 2.0  # Same continent is less ideal but still useful
                
                # Prefer regions with similar services
                service_overlap = len(set(primary_info.available_services) & set(region_info.available_services))
                score += service_overlap * 2.0
                
                # Consider compliance overlap
                compliance_overlap = len(set(primary_info.compliance_certifications) & set(region_info.compliance_certifications))
                score += compliance_overlap * 1.0
                
                # Consider cost (lower cost multiplier is better for DR)
                cost_score = (2.0 - region_info.cost_multiplier) * 1.0
                score += cost_score
                
                # Check health
                health_score = self._get_region_health_score(region)
                score += health_score
                
                candidates.append((region, score))
            
            # Sort by score and return top candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [region for region, _ in candidates[:max_regions]]
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report for multi-region deployment."""
        with self._lock:
            report = {
                "timestamp": time.time(),
                "total_regions": len(self._regions),
                "active_policies": len(self._residency_policies),
                "regional_compliance": {},
                "policy_coverage": {},
                "recommendations": []
            }
            
            # Regional compliance summary
            for region, region_info in self._regions.items():
                report["regional_compliance"][region.value] = {
                    "certifications": region_info.compliance_certifications,
                    "quantum_available": region_info.quantum_backends_available,
                    "applicable_policies": [
                        p.name for p in self._residency_policies 
                        if region in p.applicable_regions
                    ]
                }
            
            # Policy coverage analysis
            for policy in self._residency_policies:
                coverage = {
                    "applicable_regions": [r.value for r in policy.applicable_regions],
                    "allowed_regions": [r.value for r in policy.allowed_target_regions],
                    "backup_regions": [r.value for r in policy.backup_allowed_regions],
                    "data_categories": policy.data_categories
                }
                report["policy_coverage"][policy.name] = coverage
            
            # Generate recommendations
            # Check if EU regions have GDPR coverage
            eu_regions = [r for r in self._regions.keys() 
                         if self._regions[r].continent == "Europe"]
            gdpr_policies = [p for p in self._residency_policies 
                           if "GDPR" in p.compliance_requirements]
            
            if eu_regions and not gdpr_policies:
                report["recommendations"].append(
                    "Consider adding GDPR compliance policies for EU regions"
                )
            
            # Check for single points of failure
            for policy in self._residency_policies:
                if len(policy.allowed_target_regions) == 1:
                    report["recommendations"].append(
                        f"Policy '{policy.name}' has only one allowed region - "
                        f"consider adding backup regions for resilience"
                    )
            
            return report


# Global multi-region manager instance
_multi_region_manager: Optional[MultiRegionManager] = None
_region_lock = threading.Lock()


def get_multi_region_manager() -> MultiRegionManager:
    """Get the global multi-region manager."""
    global _multi_region_manager
    
    with _region_lock:
        if _multi_region_manager is None:
            _multi_region_manager = MultiRegionManager()
        return _multi_region_manager


def get_optimal_regions(**kwargs) -> List[Region]:
    """Get optimal regions using global manager."""
    return get_multi_region_manager().get_optimal_regions(**kwargs)


def validate_data_placement(
    source_region: Region,
    target_region: Region,
    data_categories: List[str]
) -> Dict[str, Any]:
    """Validate data placement using global manager."""
    return get_multi_region_manager().validate_data_placement(
        source_region, target_region, data_categories
    )


__all__ = [
    "Region",
    "DataResidencyRule",
    "RegionInfo",
    "DataResidencyPolicy",
    "MultiRegionManager",
    "get_multi_region_manager",
    "get_optimal_regions",
    "validate_data_placement"
]