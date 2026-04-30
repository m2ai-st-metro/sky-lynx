"""Main analyzer module for Sky-Lynx.

Entry point for weekly analysis of Claude Code usage insights.

Usage:
    python -m sky_lynx.analyzer [--dry-run]
    python -m sky_lynx.analyzer --auto-apply
    python -m sky_lynx.analyzer --rollback latest
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import __version__
from .agent_effectiveness_tracker import (
    build_agent_effectiveness_digest,
    run_agent_effectiveness_evaluation,
)
from .auto_applicator import auto_apply_recommendations, rollback
from .claude_client import AnalysisResult, analyze_insights
from .cost_reader import build_cost_digest, load_cost_data
from .effectiveness_tracker import (
    build_effectiveness_digest,
    run_effectiveness_evaluation,
)
from .ideaforge_reader import build_ideaforge_digest, load_ideaforge_data
from .insights_parser import (
    TrendAnalysis,
    load_weekly_analysis,
)
from .linear_writer import create_linear_issues
from .metroplex_reader import build_pipeline_health_digest, load_metroplex_data
from .mission_reader import build_mission_digest, load_mission_data
from .outcome_reader import build_outcome_digest, load_outcome_records
from .pattern_aggregator import aggregate_patterns
from .pr_drafter import create_draft_pr
from .preference_reader import build_preference_digest, load_preference_data
from .proposal_tracker import ProposalTracker
from .report_writer import write_weekly_report
from .research_reader import build_research_digest, load_research_signals
from .skill_reader import build_skill_digest, load_skill_data
from .starscream_reader import build_starscream_digest, load_starscream_data
from .taste_reader import build_taste_digest, load_taste_data
from .telemetry_reader import build_telemetry_digest, load_telemetry_data
from .trigger_listener import _load_events as _load_skylynx_events
from .model_audit_reader import build_model_audit_digest, load_model_audit_data

# Load environment variables
load_dotenv()

# Also try shared env file
shared_env = Path.home() / ".env.shared"
if shared_env.exists():
    load_dotenv(shared_env)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_metrics_summary(analysis: TrendAnalysis) -> str:
    """Format trend analysis into a summary string for Claude.

    Args:
        analysis: TrendAnalysis from insights parser

    Returns:
        Formatted string summary
    """
    current = analysis.current
    lines = [
        f"**Period**: {current.period_start.strftime('%Y-%m-%d')} to {current.period_end.strftime('%Y-%m-%d')}",
        f"**Total Sessions**: {current.total_sessions}",
    ]

    # Add trend if available
    if analysis.previous:
        trend_symbol = "+" if analysis.session_count_change >= 0 else ""
        lines.append(f"**Session Trend**: {trend_symbol}{analysis.session_count_change:.1f}%")

    # Outcomes
    if current.outcomes:
        lines.append("")
        lines.append("**Outcomes**:")
        for outcome, count in current.outcomes.most_common():
            pct = (count / current.total_sessions * 100) if current.total_sessions > 0 else 0
            lines.append(f"  - {outcome}: {count} ({pct:.0f}%)")

    # Satisfaction
    if current.satisfaction:
        lines.append("")
        lines.append("**Satisfaction**:")
        for level, count in current.satisfaction.most_common():
            lines.append(f"  - {level}: {count}")
        if analysis.satisfaction_trend:
            lines.append(f"  - Trend: {analysis.satisfaction_trend}")

    # Helpfulness
    if current.helpfulness:
        lines.append("")
        lines.append("**Claude Helpfulness**:")
        for level, count in current.helpfulness.most_common():
            lines.append(f"  - {level}: {count}")

    # Session types
    if current.session_types:
        lines.append("")
        lines.append("**Session Types**:")
        for stype, count in current.session_types.most_common():
            lines.append(f"  - {stype}: {count}")

    # Friction (most important)
    if current.friction_counts:
        lines.append("")
        lines.append("**Friction Counts**:")
        total_friction = sum(current.friction_counts.values())
        for ftype, count in current.friction_counts.most_common():
            lines.append(f"  - {ftype}: {count}")
        lines.append(f"  - **Total**: {total_friction}")

        if analysis.previous:
            trend_symbol = "+" if analysis.friction_change >= 0 else ""
            lines.append(f"  - Friction Trend: {trend_symbol}{analysis.friction_change:.1f}%")

    # Goal categories
    if current.goal_categories:
        lines.append("")
        lines.append("**Goal Categories**:")
        for category, count in current.goal_categories.most_common(5):
            lines.append(f"  - {category}: {count}")

    # Primary successes
    if current.primary_successes:
        lines.append("")
        lines.append("**Primary Successes**:")
        for success, count in current.primary_successes.most_common(5):
            lines.append(f"  - {success}: {count}")

    return "\n".join(lines)


def _load_agent_context() -> str | None:
    """Load agent registry context: registry.yaml metadata for all agents.

    Returns formatted digest string, or None if not available.
    """
    agents_dir = Path.home() / "projects" / "st-agent-registry" / "agents"
    if not agents_dir.exists():
        return None

    parts = []
    for agent_dir in sorted(agents_dir.iterdir()):
        registry_file = agent_dir / "registry.yaml"
        if not registry_file.exists():
            continue
        try:
            with open(registry_file) as f:
                reg = yaml.safe_load(f)
            agent_id = reg.get("agent_id", agent_dir.name)
            agent_type = reg.get("type", "unknown")
            learning = reg.get("learning", {})
            patches_applied = learning.get("total_patches_applied", 0)
            patches_proposed = learning.get("total_patches_proposed", 0)
            eff_score = learning.get("effectiveness_score")
            last_patch = learning.get("last_patch_at")

            line = f"  - **{agent_id}** (type: {agent_type})"
            line += f" — patches: {patches_applied} applied / {patches_proposed} proposed"
            if eff_score is not None:
                line += f", effectiveness: {eff_score:.3f}"
            if last_patch:
                line += f", last patch: {last_patch}"
            parts.append(line)
        except Exception:
            continue

    if not parts:
        return None

    return "## Agent Registry Context\n\n" + "\n".join(parts)


def run_analysis(
    dry_run: bool = False,
    scope: str = "full",
    trigger_context: str | None = None,
) -> tuple[TrendAnalysis, AnalysisResult]:
    """Run the analysis pipeline.

    Args:
        dry_run: If True, skip API calls
        scope: "full" for all 15 data sources, "pipeline" for scoped analysis
            (metroplex, outcome, ideaforge only)
        trigger_context: If set, included in Claude prompt to explain why
            this analysis was triggered reactively

    Returns:
        Tuple of (TrendAnalysis, AnalysisResult)
    """
    logger.info("Loading usage insights...")
    trend_analysis = load_weekly_analysis()

    if trend_analysis.current.total_sessions == 0:
        logger.warning("No sessions found for this week. Using available data.")

    logger.info(f"Found {trend_analysis.current.total_sessions} sessions this week")

    # Format for Claude
    metrics_summary = format_metrics_summary(trend_analysis)
    friction_details = trend_analysis.current.friction_details

    logger.info(f"Friction details: {len(friction_details)}")

    # Aggregate failure patterns from the shared skylynx-events stream
    # (Metroplex, CMD, ...). Runs on every analysis — scheduled + reactive —
    # so the failure_patterns table stays warm without needing its own cron.
    try:
        pipeline_events = _load_skylynx_events(
            Path(
                os.environ.get(
                    "SKYLYNX_EVENTS_DIR",
                    str(Path.home() / ".local" / "share" / "skylynx-events"),
                )
            )
        )
        patterns_touched = aggregate_patterns(pipeline_events)
        if patterns_touched:
            logger.info("Aggregated %d failure pattern rows", patterns_touched)
        else:
            logger.info("No failure patterns to aggregate from event stream")
    except Exception as e:
        logger.warning(f"Could not aggregate failure patterns: {e}")

    # Load outcome records from Snow-Town
    outcome_digest = None
    try:
        outcome_records = load_outcome_records()
        if outcome_records:
            outcome_digest = build_outcome_digest(outcome_records)
            logger.info(f"Loaded {len(outcome_records)} outcome records for analysis")
        else:
            logger.info("No outcome records available yet")
    except Exception as e:
        logger.warning(f"Could not load outcome records: {e}")

    # Load IdeaForge market signals
    ideaforge_digest = None
    try:
        ideaforge_data = load_ideaforge_data()
        if ideaforge_data:
            ideaforge_digest = build_ideaforge_digest(ideaforge_data)
            logger.info(f"Loaded IdeaForge data: {ideaforge_data.get('total_signals', 0)} signals, {ideaforge_data.get('total_ideas', 0)} ideas")
        else:
            logger.info("No IdeaForge data available")
    except Exception as e:
        logger.warning(f"Could not load IdeaForge data: {e}")

    # Sources below are skipped in "pipeline" scope (reactive triggers)
    research_digest = None
    taste_digest = None
    telemetry_digest = None

    if scope == "full":
        # Load research signals from Snow-Town
        try:
            research_data = load_research_signals()
            if research_data and research_data.get("total_signals", 0) > 0:
                research_digest = build_research_digest(research_data)
                logger.info(f"Loaded research signals: {research_data.get('total_signals', 0)} total")
            else:
                logger.info("No research signals available yet")
        except Exception as e:
            logger.warning(f"Could not load research signals: {e}")

        # Load taste profile delta
        try:
            taste_data = load_taste_data()
            if taste_data:
                taste_digest = build_taste_digest(taste_data)
                logger.info(f"Loaded taste delta from {taste_data['report_date']}")
            else:
                logger.info("No taste profile data available yet")
        except Exception as e:
            logger.warning(f"Could not load taste data: {e}")

        # Load Data (ClaudeClaw) telemetry
        try:
            telemetry_data = load_telemetry_data()
            if telemetry_data:
                telemetry_digest = build_telemetry_digest(telemetry_data)
                logger.info(f"Loaded telemetry data: {telemetry_data.get('total_events', 0)} events")
            else:
                logger.info("No telemetry data available from Data")
        except Exception as e:
            logger.warning(f"Could not load telemetry data: {e}")

    # Load Metroplex pipeline health metrics
    pipeline_health_digest = None
    try:
        metroplex_data = load_metroplex_data()
        if metroplex_data:
            pipeline_health_digest = build_pipeline_health_digest(metroplex_data)
            logger.info(
                "Loaded Metroplex data: %d builds, %d triage decisions",
                metroplex_data.get("build_total", 0),
                metroplex_data.get("triage_total", 0),
            )
        else:
            logger.info("No Metroplex data available")
    except Exception as e:
        logger.warning(f"Could not load Metroplex data: {e}")

    # Remaining sources skipped in "pipeline" scope
    preference_digest = None
    mission_digest = None
    cost_digest = None
    skill_digest = None
    starscream_digest = None
    agent_context_digest = None
    effectiveness_digest = None
    agent_effectiveness_digest = None
    model_audit_digest = None

    if scope == "full":
        # Load ClaudeClaw preference profile
        try:
            preference_data = load_preference_data()
            if preference_data:
                preference_digest = build_preference_digest(preference_data)
                logger.info(f"Loaded preference data: {preference_data.get('total', 0)} preferences")
            else:
                logger.info("No preference data available")
        except Exception as e:
            logger.warning(f"Could not load preference data: {e}")

        # Load ClaudeClaw mission performance
        try:
            mission_data = load_mission_data()
            if mission_data:
                mission_digest = build_mission_digest(mission_data)
                logger.info(f"Loaded mission data: {mission_data.get('total_missions', 0)} missions")
            else:
                logger.info("No mission data available")
        except Exception as e:
            logger.warning(f"Could not load mission data: {e}")

        # Load ClaudeClaw token costs
        try:
            cost_data = load_cost_data()
            if cost_data:
                cost_digest = build_cost_digest(cost_data)
                logger.info(f"Loaded cost data: ${cost_data.get('total_cost', 0):.2f} this week")
            else:
                logger.info("No cost data available")
        except Exception as e:
            logger.warning(f"Could not load cost data: {e}")

        # Load skill inventory and usage
        try:
            skill_data = load_skill_data()
            if skill_data:
                skill_digest = build_skill_digest(skill_data)
                logger.info(f"Loaded skill data: {skill_data.get('total_deployed', 0)} skills deployed")
            else:
                logger.info("No skill data available")
        except Exception as e:
            logger.warning(f"Could not load skill data: {e}")

        # Load Starscream LinkedIn analytics
        try:
            starscream_data = load_starscream_data()
            if starscream_data:
                starscream_digest = build_starscream_digest(starscream_data)
                logger.info(
                    "Loaded Starscream data: %d posts, %d followers",
                    starscream_data.get("total_posts", 0),
                    starscream_data.get("current_followers", 0),
                )
            else:
                logger.info("No Starscream analytics data available")
        except Exception as e:
            logger.warning(f"Could not load Starscream analytics: {e}")

        # Load agent registry context (registry.yaml for all agents)
        try:
            agent_context_digest = _load_agent_context()
            if agent_context_digest:
                logger.info("Loaded agent registry context for analysis")
            else:
                logger.info("No agent registry context available")
        except Exception as e:
            logger.warning(f"Could not load agent registry context: {e}")

        # Evaluate effectiveness of past recommendations (before analysis, so results inform it)
        if not dry_run:
            try:
                eval_results = run_effectiveness_evaluation()
                if eval_results:
                    logger.info(f"Evaluated {len(eval_results)} past recommendations")
            except Exception as e:
                logger.warning(f"Could not run effectiveness evaluation: {e}")

        try:
            effectiveness_digest = build_effectiveness_digest()
            if effectiveness_digest:
                logger.info("Loaded effectiveness digest for analysis context")
            else:
                logger.info("No past effectiveness data available yet")
        except Exception as e:
            logger.warning(f"Could not build effectiveness digest: {e}")

        # Evaluate effectiveness of past agent patches
        if not dry_run:
            try:
                agent_eval_results = run_agent_effectiveness_evaluation()
                if agent_eval_results:
                    logger.info(f"Evaluated {len(agent_eval_results)} past agent patches")
            except Exception as e:
                logger.warning(f"Could not run agent effectiveness evaluation: {e}")

        try:
            agent_effectiveness_digest = build_agent_effectiveness_digest()
            if agent_effectiveness_digest:
                logger.info("Loaded agent effectiveness digest for analysis context")
            else:
                logger.info("No agent effectiveness data available yet")
        except Exception as e:
            logger.warning(f"Could not build agent effectiveness digest: {e}")

        # Run model audit benchmarks (monthly cadence -- skip if last run < 30 days)
        try:
            model_audit_data = load_model_audit_data()
            if model_audit_data:
                model_audit_digest = build_model_audit_digest(model_audit_data)
                logger.info(
                    "Model audit: %d benchmarks, %d/%d checks passed, %d critical failures",
                    model_audit_data.get("total_benchmarks", 0),
                    model_audit_data.get("total_pass", 0),
                    model_audit_data.get("total_checks", 0),
                    model_audit_data.get("critical_failures", 0),
                )
            else:
                logger.info("No model audit data (runner not available or no results)")
        except Exception as e:
            logger.warning(f"Could not run model audit: {e}")

    # Inject trigger context into metrics summary for reactive runs
    if trigger_context:
        metrics_summary = (
            f"## REACTIVE TRIGGER (event-driven, not scheduled)\n"
            f"**Trigger reason**: {trigger_context}\n"
            f"**Analysis scope**: pipeline (Metroplex + IdeaForge + outcomes only)\n"
            f"Focus recommendations on pipeline health and build success.\n\n"
            + metrics_summary
        )

    # Run Claude analysis
    logger.info("Running Claude analysis..." if not dry_run else "Dry run - skipping API call")
    analysis_result = analyze_insights(
        metrics_summary=metrics_summary,
        friction_details=friction_details,
        dry_run=dry_run,
        outcome_digest=outcome_digest,
        ideaforge_digest=ideaforge_digest,
        research_digest=research_digest,
        telemetry_digest=telemetry_digest,
        taste_digest=taste_digest,
        effectiveness_digest=effectiveness_digest,
        pipeline_health_digest=pipeline_health_digest,
        preference_digest=preference_digest,
        mission_digest=mission_digest,
        skill_digest=skill_digest,
        starscream_digest=starscream_digest,
        cost_digest=cost_digest,
        agent_context_digest=agent_context_digest,
        agent_effectiveness_digest=agent_effectiveness_digest,
        model_audit_digest=model_audit_digest,
    )

    logger.info(f"Got {len(analysis_result.recommendations)} recommendations")

    return trend_analysis, analysis_result



def _run_agent_upgrader() -> None:
    """Trigger ST Records agent upgrader for pending agent recs.

    Runs the upgrader as a subprocess to keep systems decoupled.
    """
    upgrader_path = (
        Path.home() / "projects" / "st-records" / "scripts" / "agent_upgrader.py"
    )
    if not upgrader_path.exists():
        logger.warning(f"Agent upgrader not found at {upgrader_path}")
        return

    logger.info("Triggering agent upgrader for agent-targeted recommendations...")
    try:
        result = subprocess.run(
            [sys.executable, str(upgrader_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Agent upgrader completed successfully")
        else:
            logger.warning(
                f"Agent upgrader exited with code {result.returncode}: "
                f"{result.stderr[:200]}"
            )
    except subprocess.TimeoutExpired:
        logger.warning("Agent upgrader timed out after 120s")
    except Exception as e:
        logger.warning(f"Could not run agent upgrader: {e}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Sky-Lynx: Continuous Improvement Agent for Claude Code"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default analyze command (also runs with no subcommand)
    analyze_parser = subparsers.add_parser("analyze", help="Run weekly analysis (default)")
    analyze_parser.add_argument("--dry-run", action="store_true", help="Skip API calls and PR creation")
    analyze_parser.add_argument("--no-pr", action="store_true", help="Generate report but skip PR creation")
    analyze_parser.add_argument("--auto-apply", action="store_true", help="Auto-apply eligible high-confidence rules")
    analyze_parser.add_argument("--rollback", nargs="?", const="latest", default=None, metavar="BACKUP",
                                help="Rollback ~/CLAUDE.md to a backup (default: latest)")

    # Manifest refresh command
    refresh_parser = subparsers.add_parser("refresh-manifests", help="Refresh project.json health scores (daily pulse)")
    refresh_parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")

    # Proposal management commands
    subparsers.add_parser("check-proposals", help="Check for unresolved proposals and squawk if overdue")
    subparsers.add_parser("list-proposals", help="List pending pipeline config proposals")

    apply_parser = subparsers.add_parser("apply-proposal", help="Accept a pipeline config proposal")
    apply_parser.add_argument("proposal_id", type=int, help="Proposal ID to accept")

    reject_parser = subparsers.add_parser("reject-proposal", help="Reject a pipeline config proposal")
    reject_parser.add_argument("proposal_id", type=int, help="Proposal ID to reject")

    # Event-driven trigger check (Phase F)
    trigger_parser = subparsers.add_parser("check-triggers", help="Check for pipeline events and run scoped analysis if triggered")
    trigger_parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    trigger_parser.add_argument("--force", action="store_true", help="Ignore cooldown")
    trigger_parser.add_argument("--auto-apply", action="store_true", help="Auto-apply eligible rules")
    trigger_parser.add_argument("--no-pr", action="store_true", help="Skip PR creation")
    trigger_parser.add_argument("--cooldown-hours", type=float, default=12.0, help="Cooldown period in hours (default: 12)")

    # Legacy flags (for backward compatibility when no subcommand given)
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-pr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--auto-apply", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rollback", nargs="?", const="latest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"sky-lynx {__version__}")

    args = parser.parse_args()

    # Handle manifest refresh (no analysis needed)
    if args.command == "refresh-manifests":
        from .manifest_refresh import run_refresh
        results = run_refresh(dry_run=args.dry_run)
        logger.info(
            "Manifest refresh: %d/%d updated, %d stale, %d errors",
            results["updated"], results["total"],
            len(results["stale"]), len(results["errors"]),
        )
        if results["stale"]:
            logger.info("Stale projects: %s", ", ".join(results["stale"]))
        return 0

    # Handle proposal subcommands (no analysis needed)
    if args.command == "check-proposals":
        tracker = ProposalTracker()
        squawked = tracker.check_and_squawk()
        tracker.close()
        logger.info("Squawked about %d overdue proposals", squawked)
        return 0

    if args.command == "list-proposals":
        tracker = ProposalTracker()
        pending = tracker.get_pending()
        tracker.close()
        if not pending:
            print("No pending proposals.")
            return 0
        print(f"{'ID':>4}  {'Status':>10}  {'Squawks':>7}  Parameter")
        print("-" * 60)
        for p in pending:
            print(f"{p['id']:>4}  {p['status']:>10}  {p['squawk_count']:>7}  {p['parameter']}")
            print(f"      Current: {p['current_value']}  →  Proposed: {p['proposed_value']}")
            print(f"      Rationale: {p['rationale'][:80]}")
            print()
        return 0

    if args.command == "apply-proposal":
        tracker = ProposalTracker()
        if tracker.accept(args.proposal_id):
            print(f"Proposal #{args.proposal_id} accepted.")
        else:
            print(f"Proposal #{args.proposal_id} not found or already resolved.")
        tracker.close()
        return 0

    if args.command == "reject-proposal":
        tracker = ProposalTracker()
        if tracker.reject(args.proposal_id):
            print(f"Proposal #{args.proposal_id} rejected.")
        else:
            print(f"Proposal #{args.proposal_id} not found or already resolved.")
        tracker.close()
        return 0

    # Handle check-triggers (Phase F event-driven)
    if args.command == "check-triggers":
        from .trigger_listener import (
            evaluate_triggers, check_cooldown, record_trigger, cleanup_events,
        )

        trigger = evaluate_triggers()
        if trigger is None:
            logger.info("No trigger conditions met")
            cleanup_events()
            return 0

        if not args.force and not check_cooldown(cooldown_hours=args.cooldown_hours):
            logger.info("Trigger conditions met but cooldown active: %s", trigger.reason)
            return 0

        logger.info("TRIGGERED: %s", trigger.reason)
        logger.info("=" * 60)
        logger.info(f"Sky-Lynx v{__version__} REACTIVE RUN at {datetime.now().isoformat()}")
        logger.info("=" * 60)

        try:
            trend_analysis, analysis_result = run_analysis(
                dry_run=args.dry_run,
                scope=trigger.scope,
                trigger_context=trigger.reason,
            )
            record_trigger()
            cleanup_events()

            # Reuse existing output pipeline
            auto_applied_titles = set()
            if args.auto_apply and not args.dry_run:
                try:
                    applied = auto_apply_recommendations(
                        analysis_result.recommendations,
                        session_id=f"sky-lynx-trigger-{datetime.now().strftime('%Y-%m-%d-%H%M')}",
                    )
                    auto_applied_titles = {r.title for r in applied}
                    logger.info("Auto-applied %d rules", len(applied))
                except Exception as e:
                    logger.warning(f"Auto-apply failed: {e}")

            # Write report
            report_path = write_weekly_report(trend_analysis, analysis_result)
            logger.info(f"Report written to: {report_path}")

            # Create PR (unless suppressed)
            if not args.no_pr and not args.dry_run:
                try:
                    pr_url = create_draft_pr(
                        analysis_result,
                        exclude_titles=auto_applied_titles,
                    )
                    if pr_url:
                        logger.info(f"Draft PR created: {pr_url}")
                except Exception as e:
                    logger.warning(f"PR creation failed: {e}")

            logger.info("Reactive analysis complete")
            return 0
        except Exception as e:
            logger.error(f"Reactive analysis failed: {e}", exc_info=True)
            return 1

    logger.info("=" * 60)
    logger.info(f"Sky-Lynx v{__version__} starting at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    try:
        # Handle --rollback mode (standalone, no analysis needed)
        if args.rollback is not None:
            logger.info(f"Rolling back CLAUDE.md to: {args.rollback}")
            success = rollback(args.rollback)
            if success:
                logger.info("Rollback successful")
                return 0
            else:
                logger.error("Rollback failed")
                return 1

        # Run analysis
        trend_analysis, analysis_result = run_analysis(dry_run=args.dry_run)

        # Auto-apply eligible recommendations (before report/PR so report can include results)
        auto_apply_results = None
        auto_applied_titles: set[str] = set()
        if args.auto_apply:
            session_id = f"sky-lynx-{datetime.now().strftime('%Y-%m-%d')}"
            logger.info("Running auto-apply for eligible recommendations...")
            auto_apply_results = auto_apply_recommendations(
                analysis_result.recommendations,
                session_id=session_id,
                dry_run=args.dry_run,
            )
            auto_applied_titles = {
                r.title for r in auto_apply_results if r.applied
            }
            if auto_applied_titles:
                logger.info(f"Auto-applied {len(auto_applied_titles)} rules to ~/CLAUDE.md")

        # Create proposals for pipeline config recommendations
        pipeline_recs = [
            r for r in analysis_result.recommendations
            if r.target_system == "pipeline"
        ]
        if pipeline_recs and not args.dry_run:
            tracker = ProposalTracker()
            auto_accepted = 0
            for rec in pipeline_recs:
                proposal_id = tracker.propose(
                    parameter=rec.title,
                    current_value="(current)",
                    proposed_value=rec.suggested_change,
                    rationale=rec.evidence,
                )
                # Auto-accept pipeline proposals with sufficient confidence:
                # - High priority always qualifies (with high reversibility)
                # - Medium priority qualifies only if highly reversible
                _eligible = (
                    rec.evidence
                    and len(rec.evidence) >= 40
                    and (
                        (rec.priority == "high" and rec.reversibility in ("high", "medium"))
                        or (rec.priority == "medium" and rec.reversibility == "high")
                    )
                )
                if _eligible:
                    tracker.accept(proposal_id)
                    auto_accepted += 1
                    logger.info(
                        "Auto-accepted pipeline proposal #%d: %s",
                        proposal_id, rec.title,
                    )
            tracker.close()
            logger.info(
                "Created %d pipeline proposals (%d auto-accepted)",
                len(pipeline_recs), auto_accepted,
            )

        # Write report (pass auto-apply results for report section)
        logger.info("Writing weekly report...")
        report_path = write_weekly_report(
            trend_analysis,
            analysis_result,
            auto_apply_results=auto_apply_results,
        )
        logger.info(f"Report saved to: {report_path}")

        # Create PR (unless --no-pr or --dry-run), excluding auto-applied recs
        if not args.no_pr and not args.dry_run:
            logger.info("Creating draft PR...")
            pr_url = create_draft_pr(
                analysis_result,
                exclude_titles=auto_applied_titles,
            )
            if pr_url:
                logger.info(f"Draft PR created: {pr_url}")
            else:
                logger.info("No changes to propose - skipping PR creation")
        else:
            logger.info("Skipping PR creation (--dry-run or --no-pr)")

        # Create Linear issues for visibility (10a)
        session_id = f"sky-lynx-{datetime.now().strftime('%Y-%m-%d')}"
        try:
            created_issues = create_linear_issues(
                analysis_result.recommendations,
                session_id=session_id,
                dry_run=args.dry_run,
            )
            if created_issues:
                logger.info(
                    f"Created {len(created_issues)} Linear issues: "
                    f"{', '.join(created_issues)}"
                )
        except Exception as e:
            logger.warning(f"Could not create Linear issues: {e}")

        # Write ClaudeClaw-targeted recommendations as JSON files (10c)
        try:
            from .claudeclaw_writer import write_claudeclaw_recommendations

            written_files = write_claudeclaw_recommendations(
                analysis_result.recommendations
            )
            if written_files:
                logger.info(
                    f"Wrote {len(written_files)} ClaudeClaw recommendations"
                )
        except Exception as e:
            logger.warning(f"Could not write ClaudeClaw recommendations: {e}")

        # Trigger agent upgrader for agent-targeted recs (10d)
        agent_recs = [
            r for r in analysis_result.recommendations
            if r.target_system == "agent"
        ]
        if agent_recs and not args.dry_run:
            _run_agent_upgrader()

        logger.info("=" * 60)
        logger.info("Sky-Lynx analysis complete")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
