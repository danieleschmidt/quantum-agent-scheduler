#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Scheduler
Manages continuous execution of value-driven SDLC improvements.
"""

import subprocess
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from value_engine import ValueDiscoveryEngine, ValueItem
from autonomous_executor import AutonomousExecutor


@dataclass
class ScheduleConfig:
    """Configuration for autonomous scheduling."""
    immediate_after_merge: bool = True
    hourly_security_scan: bool = True
    daily_comprehensive_analysis: bool = True
    weekly_deep_review: bool = True
    monthly_strategic_review: bool = True
    
    max_concurrent_tasks: int = 1
    execution_timeout_minutes: int = 30
    min_interval_minutes: int = 15


class AutonomousScheduler:
    """Autonomous SDLC scheduler with value-driven execution."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config = ScheduleConfig()
        self.executor = AutonomousExecutor(config_path)
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        
        self.metrics_file = Path(".terragon/scheduler-metrics.json")
        self.last_execution_file = Path(".terragon/last-execution.json")
        
        self.metrics = self._load_metrics()
        self.last_execution = self._load_last_execution()
    
    def _load_metrics(self) -> Dict:
        """Load scheduler metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            "total_cycles": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_value_delivered": 0.0,
            "average_cycle_time": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_last_execution(self) -> Dict:
        """Load last execution timestamp."""
        if self.last_execution_file.exists():
            with open(self.last_execution_file, 'r') as f:
                return json.load(f)
        return {
            "last_immediate": None,
            "last_hourly": None,
            "last_daily": None,
            "last_weekly": None,
            "last_monthly": None
        }
    
    def _save_metrics(self):
        """Save scheduler metrics."""
        self.metrics["last_updated"] = datetime.now().isoformat()
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _save_last_execution(self):
        """Save last execution timestamp."""
        self.last_execution_file.parent.mkdir(exist_ok=True)
        with open(self.last_execution_file, 'w') as f:
            json.dump(self.last_execution, f, indent=2)
    
    def should_execute_immediate(self) -> bool:
        """Check if immediate execution should run (after PR merge)."""
        if not self.config.immediate_after_merge:
            return False
        
        # Check if there was a recent merge by looking at git log
        try:
            result = subprocess.run([
                'git', 'log', '--oneline', '--since=10 minutes ago', '--grep=Merge'
            ], capture_output=True, text=True)
            
            recent_merges = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            if recent_merges:
                last_immediate = self.last_execution.get("last_immediate")
                if not last_immediate:
                    return True
                
                last_time = datetime.fromisoformat(last_immediate)
                if datetime.now() - last_time > timedelta(minutes=self.config.min_interval_minutes):
                    return True
        
        except subprocess.CalledProcessError:
            pass
        
        return False
    
    def should_execute_hourly(self) -> bool:
        """Check if hourly execution should run."""
        if not self.config.hourly_security_scan:
            return False
        
        last_hourly = self.last_execution.get("last_hourly")
        if not last_hourly:
            return True
        
        last_time = datetime.fromisoformat(last_hourly)
        return datetime.now() - last_time >= timedelta(hours=1)
    
    def should_execute_daily(self) -> bool:
        """Check if daily execution should run."""
        if not self.config.daily_comprehensive_analysis:
            return False
        
        last_daily = self.last_execution.get("last_daily")
        if not last_daily:
            return True
        
        last_time = datetime.fromisoformat(last_daily)
        return datetime.now() - last_time >= timedelta(days=1)
    
    def should_execute_weekly(self) -> bool:
        """Check if weekly execution should run."""
        if not self.config.weekly_deep_review:
            return False
        
        last_weekly = self.last_execution.get("last_weekly")
        if not last_weekly:
            return True
        
        last_time = datetime.fromisoformat(last_weekly)
        return datetime.now() - last_time >= timedelta(weeks=1)
    
    def should_execute_monthly(self) -> bool:
        """Check if monthly execution should run."""
        if not self.config.monthly_strategic_review:
            return False
        
        last_monthly = self.last_execution.get("last_monthly")
        if not last_monthly:
            return True
        
        last_time = datetime.fromisoformat(last_monthly)
        return datetime.now() - last_time >= timedelta(days=30)
    
    def execute_immediate_cycle(self) -> bool:
        """Execute immediate post-merge value cycle."""
        print("🚀 Executing immediate post-merge value cycle...")
        
        start_time = datetime.now()
        success = False
        
        try:
            # Focus on critical security and build fixes
            value_items = self.discovery_engine.discover_value_opportunities()
            critical_items = [
                item for item in value_items 
                if item.category.value in ['security', 'infrastructure'] 
                and item.composite_score >= 15
            ]
            
            if critical_items:
                executed_item = self.executor.execute_next_best_value()
                success = executed_item is not None
            else:
                print("ℹ️  No critical items found for immediate execution")
                success = True
        
        except Exception as e:
            print(f"❌ Immediate cycle failed: {e}")
        
        finally:
            self.last_execution["last_immediate"] = datetime.now().isoformat()
            self._update_cycle_metrics("immediate", start_time, success)
        
        return success
    
    def execute_hourly_cycle(self) -> bool:
        """Execute hourly security and vulnerability cycle."""
        print("🔒 Executing hourly security scan cycle...")
        
        start_time = datetime.now()
        success = False
        
        try:
            # Focus on security vulnerabilities and dependency issues
            value_items = self.discovery_engine.discover_value_opportunities()
            security_items = [
                item for item in value_items 
                if item.category.value in ['security', 'dependency']
                and item.composite_score >= 10
            ]
            
            if security_items:
                # Execute top security item
                executed_item = self.executor.execute_next_best_value()
                success = executed_item is not None
            else:
                print("ℹ️  No security items found for hourly execution")
                success = True
        
        except Exception as e:
            print(f"❌ Hourly cycle failed: {e}")
        
        finally:
            self.last_execution["last_hourly"] = datetime.now().isoformat()
            self._update_cycle_metrics("hourly", start_time, success)
        
        return success
    
    def execute_daily_cycle(self) -> bool:
        """Execute daily comprehensive analysis cycle."""
        print("📊 Executing daily comprehensive analysis cycle...")
        
        start_time = datetime.now()
        success = False
        
        try:
            # Full value discovery and execution of top items
            value_items = self.discovery_engine.discover_value_opportunities()
            
            if value_items:
                # Execute multiple items based on available time
                executed_count = 0
                max_items = 3  # Execute up to 3 items daily
                
                for _ in range(max_items):
                    executed_item = self.executor.execute_next_best_value()
                    if executed_item:
                        executed_count += 1
                    else:
                        break
                
                success = executed_count > 0
                print(f"✅ Daily cycle executed {executed_count} items")
            else:
                print("ℹ️  No items found for daily execution")
                success = True
        
        except Exception as e:
            print(f"❌ Daily cycle failed: {e}")
        
        finally:
            self.last_execution["last_daily"] = datetime.now().isoformat()
            self._update_cycle_metrics("daily", start_time, success)
        
        return success
    
    def execute_weekly_cycle(self) -> bool:
        """Execute weekly deep review and optimization cycle."""  
        print("🔍 Executing weekly deep review cycle...")
        
        start_time = datetime.now()
        success = False
        
        try:
            # Deep architectural analysis and technical debt reduction
            value_items = self.discovery_engine.discover_value_opportunities()
            
            # Focus on architectural improvements and large technical debt items
            strategic_items = [
                item for item in value_items
                if item.category.value in ['technical_debt', 'performance', 'quantum_optimization']
                and item.estimated_hours >= 2  # Larger improvements
            ]
            
            if strategic_items:
                executed_count = 0
                max_items = 5  # More items for weekly cycle
                
                for _ in range(max_items):
                    executed_item = self.executor.execute_next_best_value()
                    if executed_item:
                        executed_count += 1
                    else:
                        break
                
                success = executed_count > 0
                print(f"✅ Weekly cycle executed {executed_count} strategic items")
            else:
                print("ℹ️  No strategic items found for weekly execution")
                success = True
        
        except Exception as e:
            print(f"❌ Weekly cycle failed: {e}")
        
        finally:
            self.last_execution["last_weekly"] = datetime.now().isoformat()
            self._update_cycle_metrics("weekly", start_time, success)
        
        return success
    
    def execute_monthly_cycle(self) -> bool:
        """Execute monthly strategic review and modernization cycle."""
        print("🎯 Executing monthly strategic review cycle...")
        
        start_time = datetime.now()
        success = False
        
        try:
            # Strategic value alignment and scoring model recalibration
            print("📈 Recalibrating scoring model based on execution outcomes...")
            
            # Analyze execution metrics and adjust weights
            self._recalibrate_scoring_model()
            
            # Technology stack modernization opportunities
            value_items = self.discovery_engine.discover_value_opportunities()
            
            # Focus on modernization and innovation
            innovation_items = [
                item for item in value_items
                if item.category.value in ['quantum_optimization', 'infrastructure', 'performance']
                and item.estimated_hours >= 3  # Substantial improvements
            ]
            
            if innovation_items:
                executed_count = 0
                max_items = 8  # Comprehensive monthly improvements
                
                for _ in range(max_items):
                    executed_item = self.executor.execute_next_best_value()
                    if executed_item:
                        executed_count += 1
                    else:
                        break
                
                success = executed_count > 0
                print(f"✅ Monthly cycle executed {executed_count} innovation items")
            else:
                print("ℹ️  No innovation items found for monthly execution")
                success = True
        
        except Exception as e:
            print(f"❌ Monthly cycle failed: {e}")
        
        finally:
            self.last_execution["last_monthly"] = datetime.now().isoformat()
            self._update_cycle_metrics("monthly", start_time, success)
        
        return success
    
    def _recalibrate_scoring_model(self):
        """Recalibrate scoring model based on execution outcomes."""
        # Placeholder for scoring model improvement based on historical data
        print("🔄 Analyzing historical execution data for scoring improvements...")
        
        # Load execution metrics and adjust weights based on success patterns
        # This would analyze which categories delivered the most value
        # and adjust the scoring weights accordingly
        pass
    
    def _update_cycle_metrics(self, cycle_type: str, start_time: datetime, success: bool):
        """Update metrics for execution cycle."""
        cycle_time = (datetime.now() - start_time).total_seconds()
        
        self.metrics["total_cycles"] += 1
        
        if success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update average cycle time
        if "cycle_times" not in self.metrics:
            self.metrics["cycle_times"] = []
        
        self.metrics["cycle_times"].append({
            "type": cycle_type,
            "duration": cycle_time,
            "timestamp": datetime.now().isoformat(),
            "success": success
        })
        
        # Keep only last 100 cycle times
        if len(self.metrics["cycle_times"]) > 100:
            self.metrics["cycle_times"] = self.metrics["cycle_times"][-100:]
        
        # Calculate average cycle time
        recent_times = [c["duration"] for c in self.metrics["cycle_times"][-10:]]
        self.metrics["average_cycle_time"] = sum(recent_times) / len(recent_times) if recent_times else 0.0
        
        self._save_metrics()
        self._save_last_execution()
    
    def run_scheduler_daemon(self, check_interval_minutes: int = 5):
        """Run continuous scheduler daemon."""
        print(f"🚀 Starting Terragon Autonomous SDLC Scheduler daemon...")
        print(f"⏰ Check interval: {check_interval_minutes} minutes")
        
        while True:
            try:
                current_time = datetime.now()
                print(f"\\n⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Checking execution schedules...")
                
                executed_any = False
                
                # Check each execution schedule
                if self.should_execute_immediate():
                    self.execute_immediate_cycle()
                    executed_any = True
                
                if self.should_execute_hourly():
                    self.execute_hourly_cycle()
                    executed_any = True
                
                if self.should_execute_daily():
                    self.execute_daily_cycle()
                    executed_any = True
                
                if self.should_execute_weekly():
                    self.execute_weekly_cycle()
                    executed_any = True
                
                if self.should_execute_monthly():
                    self.execute_monthly_cycle()
                    executed_any = True
                
                if not executed_any:
                    print("ℹ️  No scheduled executions needed")
                
                # Print current metrics
                success_rate = (
                    self.metrics["successful_executions"] / 
                    max(self.metrics["total_cycles"], 1)
                ) * 100
                
                print(f"📊 Scheduler Metrics:")
                print(f"   Total Cycles: {self.metrics['total_cycles']}")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Avg Cycle Time: {self.metrics['average_cycle_time']:.1f}s")
                
            except Exception as e:
                print(f"❌ Scheduler error: {e}")
            
            # Wait for next check
            print(f"⏱️  Sleeping for {check_interval_minutes} minutes...")
            time.sleep(check_interval_minutes * 60)
    
    def run_single_cycle(self) -> bool:
        """Run a single execution cycle based on current needs."""
        print("🔄 Running single autonomous execution cycle...")
        
        # Determine which cycle to run based on urgency
        if self.should_execute_immediate():
            return self.execute_immediate_cycle()
        elif self.should_execute_hourly():
            return self.execute_hourly_cycle()
        elif self.should_execute_daily():
            return self.execute_daily_cycle()
        elif self.should_execute_weekly():
            return self.execute_weekly_cycle()
        elif self.should_execute_monthly():
            return self.execute_monthly_cycle()
        else:
            # Force daily cycle if nothing else needed
            print("ℹ️  No scheduled cycles due, running daily cycle...")
            return self.execute_daily_cycle()


def main():
    """Main entry point for autonomous scheduler."""
    import sys
    
    scheduler = AutonomousScheduler()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "daemon":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            scheduler.run_scheduler_daemon(interval)
        elif command == "immediate":
            scheduler.execute_immediate_cycle()
        elif command == "hourly":
            scheduler.execute_hourly_cycle()
        elif command == "daily":
            scheduler.execute_daily_cycle()
        elif command == "weekly":
            scheduler.execute_weekly_cycle()
        elif command == "monthly":
            scheduler.execute_monthly_cycle()
        else:
            print("Usage: scheduler.py [daemon|immediate|hourly|daily|weekly|monthly]")
    else:
        scheduler.run_single_cycle()


if __name__ == "__main__":
    main()