#!/bin/bash
# Terragon Autonomous SDLC System - Main Integration Script
# Quantum Agent Scheduler Repository Enhancement

set -e

echo "🤖 Terragon Autonomous SDLC System"
echo "   Repository: quantum-agent-scheduler"
echo "   Enhancement Level: Advanced (75%+ maturity)"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Function to print section headers
print_section() {
    echo ""
    echo "================================"
    echo "$1"
    echo "================================"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_section "🔍 SYSTEM INITIALIZATION"

# Check Python availability
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python found: $PYTHON_CMD"

# Check if Terragon system is installed
if [ ! -f ".terragon/config.yaml" ]; then
    echo "❌ Error: Terragon system not found. Please run the installation first."
    exit 1
fi

echo "✅ Terragon system detected"

print_section "📊 VALUE DISCOVERY EXECUTION"

# Run value discovery engine
echo "🔍 Running autonomous value discovery..."
$PYTHON_CMD .terragon/value-engine.py

if [ $? -eq 0 ]; then
    echo "✅ Value discovery completed successfully"
    
    # Display summary from backlog
    if [ -f "AUTONOMOUS_BACKLOG.md" ]; then
        echo ""
        echo "📋 Generated Backlog Summary:"
        head -20 AUTONOMOUS_BACKLOG.md | grep -E "Total Items|Next Best Value|Composite Score"
    fi
else
    echo "⚠️  Value discovery completed with warnings"
fi

print_section "🚀 AUTONOMOUS EXECUTION"

# Check if continuous execution is requested
if [ "$1" = "--continuous" ]; then
    MAX_ITERATIONS=${2:-5}
    echo "🔄 Running continuous autonomous execution (max $MAX_ITERATIONS items)"
    
    $PYTHON_CMD .terragon/autonomous-executor.py --continuous $MAX_ITERATIONS
    
    if [ $? -eq 0 ]; then
        echo "✅ Continuous execution completed successfully"
    else
        echo "⚠️  Continuous execution completed with issues"
    fi

elif [ "$1" = "--single" ]; then
    echo "🎯 Running single autonomous execution cycle"
    
    $PYTHON_CMD .terragon/autonomous-executor.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Single execution completed successfully"
    else
        echo "⚠️  Single execution completed with issues"
    fi

elif [ "$1" = "--daemon" ]; then
    INTERVAL=${2:-5}
    echo "⏰ Starting autonomous scheduler daemon (interval: ${INTERVAL}m)"
    echo "   Press Ctrl+C to stop the daemon"
    
    $PYTHON_CMD .terragon/scheduler.py daemon $INTERVAL

else
    echo "ℹ️  Skipping autonomous execution (use --single, --continuous, or --daemon)"
    echo "   Usage: $0 [--single|--continuous N|--daemon M]"
    echo "     --single: Execute one highest-value item"
    echo "     --continuous N: Execute up to N items continuously"  
    echo "     --daemon M: Start daemon checking every M minutes"
fi

print_section "📈 SYSTEM METRICS"

# Display execution metrics if available
if [ -f ".terragon/execution-metrics.json" ]; then
    echo "📊 Execution Metrics:"
    if command_exists jq; then
        cat .terragon/execution-metrics.json | jq '{
            tasks_executed,
            success_rate,
            total_value_delivered,
            last_updated
        }'
    else
        echo "   (Install 'jq' for formatted metrics display)"
        grep -E '"tasks_executed"|"success_rate"|"total_value_delivered"' .terragon/execution-metrics.json
    fi
else
    echo "ℹ️  No execution metrics available yet"
fi

# Display scheduler metrics if available  
if [ -f ".terragon/scheduler-metrics.json" ]; then
    echo ""
    echo "⏰ Scheduler Metrics:"
    if command_exists jq; then
        cat .terragon/scheduler-metrics.json | jq '{
            total_cycles,
            successful_executions,
            average_cycle_time,
            last_updated
        }'
    else
        grep -E '"total_cycles"|"successful_executions"|"average_cycle_time"' .terragon/scheduler-metrics.json
    fi
else
    echo "ℹ️  No scheduler metrics available yet"
fi

print_section "🎯 NEXT STEPS"

echo "The Terragon Autonomous SDLC System is now operational!"
echo ""
echo "📋 To view the current backlog:"
echo "   cat AUTONOMOUS_BACKLOG.md"
echo ""
echo "🚀 To execute the next best value item:"
echo "   ./terragon-autonomous-sdlc.sh --single"
echo ""
echo "🔄 To run continuous execution:"
echo "   ./terragon-autonomous-sdlc.sh --continuous 10"
echo ""
echo "⏰ To start the autonomous daemon:"
echo "   ./terragon-autonomous-sdlc.sh --daemon 5"
echo ""
echo "📊 System configuration and metrics:"
echo "   ls -la .terragon/"
echo ""
echo "🤖 The system will continuously discover and execute the highest-value"
echo "   SDLC improvements, delivering measurable value through autonomous"
echo "   enhancement of your quantum computing development workflow."
echo ""

# Display current top 3 value items
if [ -f ".terragon/backlog.json" ]; then
    echo "🎯 Current Top 3 Value Opportunities:"
    if command_exists jq; then
        cat .terragon/backlog.json | jq -r '.items[:3] | .[] | "   \(.id): \(.title) (Score: \(.composite_score | floor))"'
    else
        echo "   (Install 'jq' to see top opportunities)"
    fi
    echo ""
fi

echo "✅ Terragon Autonomous SDLC System ready for continuous value delivery!"

exit 0