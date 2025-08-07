#!/usr/bin/env python3
"""
Integration Example: Automated Quality Report Generation in Clinical Metabolomics Oracle

This example demonstrates how to integrate the automated quality report generation
system with the existing CMO-LIGHTRAG quality validation components.

Key Integration Points:
1. Response Relevance Scoring System (CMO-LIGHTRAG-009-T02)
2. Factual Accuracy Validation (CMO-LIGHTRAG-009-T03) 
3. Performance Benchmarking Utilities (CMO-LIGHTRAG-009-T04)
4. Automated Quality Report Generation (CMO-LIGHTRAG-009-T05)

Author: Claude Code (Anthropic)
Created: August 7, 2025
Related to: CMO-LIGHTRAG-009-T05 - Integration example for automated quality reports
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from quality_report_generator import (
    QualityReportConfiguration,
    QualityReportGenerator,
    generate_quality_report,
    generate_quick_quality_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CMOQualityReportingService:
    """
    Clinical Metabolomics Oracle Quality Reporting Service
    
    This service integrates with the existing CMO-LIGHTRAG quality validation
    pipeline to provide automated quality reporting capabilities.
    """
    
    def __init__(self, output_directory: Path = None):
        """Initialize the CMO quality reporting service."""
        self.output_directory = output_directory or Path.cwd() / "cmo_quality_reports"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Default configuration for clinical metabolomics reporting
        self.default_config = QualityReportConfiguration(
            report_name="Clinical Metabolomics Oracle - Quality Validation Report",
            report_description="Automated quality assessment of LightRAG responses for clinical metabolomics applications",
            analysis_period_days=7,
            include_historical_comparison=True,
            historical_comparison_days=30,
            include_executive_summary=True,
            include_detailed_metrics=True,
            include_trend_analysis=True,
            include_performance_analysis=True,
            include_factual_accuracy_analysis=True,
            include_relevance_scoring_analysis=True,
            include_insights_and_recommendations=True,
            output_formats=['json', 'html', 'csv'],
            generate_charts=True,
            quality_score_thresholds={
                'excellent': 92.0,  # High standards for medical/clinical content
                'good': 85.0,
                'acceptable': 78.0,
                'marginal': 70.0,
                'poor': 0.0
            },
            alert_thresholds={
                'quality_decline_threshold': 8.0,   # Stricter for medical content
                'low_accuracy_threshold': 80.0,     # Higher accuracy required
                'high_error_rate_threshold': 3.0,   # Lower error tolerance
                'response_time_threshold': 2500.0   # Reasonable response time
            },
            custom_branding={
                'organization': 'Clinical Metabolomics Oracle',
                'department': 'LightRAG Quality Assurance',
                'contact': 'quality-team@cmo.org'
            }
        )
        
        logger.info(f"CMO Quality Reporting Service initialized with output directory: {self.output_directory}")
    
    async def generate_daily_quality_report(self) -> dict:
        """Generate a daily quality report for the CMO-LIGHTRAG system."""
        logger.info("Generating daily quality report...")
        
        daily_config = self.default_config
        daily_config.analysis_period_days = 1
        daily_config.report_name = f"CMO Daily Quality Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        try:
            exported_files = await generate_quality_report(
                config=daily_config,
                output_directory=self.output_directory / "daily"
            )
            
            logger.info(f"Daily quality report generated in {len(exported_files)} formats")
            return {
                'status': 'success',
                'report_type': 'daily',
                'files': exported_files,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating daily quality report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_weekly_quality_report(self) -> dict:
        """Generate a comprehensive weekly quality report."""
        logger.info("Generating weekly quality report...")
        
        weekly_config = self.default_config
        weekly_config.analysis_period_days = 7
        weekly_config.report_name = f"CMO Weekly Quality Report - Week of {datetime.now().strftime('%Y-%m-%d')}"
        weekly_config.include_trend_analysis = True
        weekly_config.include_historical_comparison = True
        
        try:
            exported_files = await generate_quality_report(
                config=weekly_config,
                output_directory=self.output_directory / "weekly"
            )
            
            logger.info(f"Weekly quality report generated in {len(exported_files)} formats")
            return {
                'status': 'success',
                'report_type': 'weekly',
                'files': exported_files,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating weekly quality report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_custom_quality_report(self, 
                                           period_start: datetime,
                                           period_end: datetime,
                                           report_name: str = None) -> dict:
        """Generate a quality report for a custom time period."""
        logger.info(f"Generating custom quality report from {period_start} to {period_end}")
        
        period_days = (period_end - period_start).days
        
        custom_config = self.default_config
        custom_config.analysis_period_days = period_days
        custom_config.report_name = report_name or f"CMO Custom Quality Report - {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}"
        
        try:
            generator = QualityReportGenerator(
                config=custom_config,
                output_directory=self.output_directory / "custom"
            )
            
            report_data = await generator.generate_quality_report(
                custom_period_start=period_start,
                custom_period_end=period_end
            )
            
            exported_files = await generator.export_report(report_data)
            
            logger.info(f"Custom quality report generated in {len(exported_files)} formats")
            return {
                'status': 'success',
                'report_type': 'custom',
                'period': {'start': period_start.isoformat(), 'end': period_end.isoformat()},
                'files': exported_files,
                'report_data': report_data,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating custom quality report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_quality_summary(self) -> dict:
        """Get a quick quality summary for monitoring dashboards."""
        logger.info("Generating quality summary for monitoring...")
        
        try:
            summary = await generate_quick_quality_summary()
            
            logger.info("Quality summary generated successfully")
            return {
                'status': 'success',
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating quality summary: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def schedule_automated_reports(self):
        """
        Example of how to schedule automated quality reports.
        In production, this would integrate with a task scheduler like Celery or APScheduler.
        """
        logger.info("Setting up automated quality report scheduling...")
        
        # This is a demonstration of how scheduled reports could work
        schedule_config = {
            'daily_reports': {
                'enabled': True,
                'time': '06:00',  # 6 AM daily
                'recipients': ['quality-team@cmo.org', 'ops-team@cmo.org']
            },
            'weekly_reports': {
                'enabled': True,
                'day': 'Monday',
                'time': '08:00',  # 8 AM Monday
                'recipients': ['management@cmo.org', 'quality-team@cmo.org']
            },
            'monthly_reports': {
                'enabled': True,
                'day': 1,  # First day of month
                'time': '09:00',
                'recipients': ['executives@cmo.org', 'quality-team@cmo.org']
            },
            'alert_thresholds': {
                'quality_decline_alert': 10.0,  # Alert if quality drops >10%
                'error_rate_alert': 5.0,        # Alert if error rate >5%
                'response_time_alert': 3000.0   # Alert if response time >3s
            }
        }
        
        logger.info("Automated reporting schedule configured:")
        for report_type, config in schedule_config.items():
            if isinstance(config, dict) and config.get('enabled'):
                logger.info(f"  - {report_type}: {config.get('time', 'N/A')}")
        
        return schedule_config


async def demo_cmo_quality_reporting():
    """Demonstrate the CMO Quality Reporting Service functionality."""
    print("="*80)
    print("CLINICAL METABOLOMICS ORACLE - QUALITY REPORTING SERVICE DEMO")
    print("="*80)
    
    # Initialize the service
    service = CMOQualityReportingService()
    
    # Demo 1: Quick quality summary
    print("\n1. Generating quick quality summary for monitoring dashboard...")
    summary_result = await service.get_quality_summary()
    
    if summary_result['status'] == 'success':
        summary = summary_result['summary']
        print(f"   ✓ Overall Health Score: {summary.get('overall_health_score', 'N/A')}/100")
        print(f"   ✓ Health Grade: {summary.get('health_grade', 'N/A')}")
        print(f"   ✓ Total Evaluations: {summary.get('total_evaluations', 'N/A')}")
    else:
        print(f"   ✗ Error: {summary_result['error']}")
    
    # Demo 2: Daily quality report
    print("\n2. Generating daily quality report...")
    daily_result = await service.generate_daily_quality_report()
    
    if daily_result['status'] == 'success':
        print(f"   ✓ Daily report generated in {len(daily_result['files'])} formats:")
        for format_type, file_path in daily_result['files'].items():
            file_size = Path(file_path).stat().st_size
            print(f"     - {format_type.upper()}: {Path(file_path).name} ({file_size:,} bytes)")
    else:
        print(f"   ✗ Error: {daily_result['error']}")
    
    # Demo 3: Weekly quality report
    print("\n3. Generating weekly quality report...")
    weekly_result = await service.generate_weekly_quality_report()
    
    if weekly_result['status'] == 'success':
        print(f"   ✓ Weekly report generated in {len(weekly_result['files'])} formats:")
        for format_type, file_path in weekly_result['files'].items():
            file_size = Path(file_path).stat().st_size
            print(f"     - {format_type.upper()}: {Path(file_path).name} ({file_size:,} bytes)")
    else:
        print(f"   ✗ Error: {weekly_result['error']}")
    
    # Demo 4: Custom period report
    print("\n4. Generating custom period quality report...")
    custom_end = datetime.now()
    custom_start = custom_end - timedelta(days=3)
    
    custom_result = await service.generate_custom_quality_report(
        period_start=custom_start,
        period_end=custom_end,
        report_name="CMO 3-Day Quality Analysis"
    )
    
    if custom_result['status'] == 'success':
        print(f"   ✓ Custom report generated for 3-day period:")
        print(f"     Period: {custom_start.strftime('%Y-%m-%d')} to {custom_end.strftime('%Y-%m-%d')}")
        print(f"     Files: {len(custom_result['files'])} formats generated")
        
        # Show insights from custom report
        report_data = custom_result['report_data']
        insights = report_data.get('insights_and_recommendations', [])
        if insights:
            print(f"     Insights: {len(insights)} quality insights generated")
            for i, insight in enumerate(insights[:3], 1):  # Show first 3
                print(f"       {i}. {insight.get('title', 'N/A')} (severity: {insight.get('severity', 'N/A')})")
    else:
        print(f"   ✗ Error: {custom_result['error']}")
    
    # Demo 5: Automated scheduling setup
    print("\n5. Setting up automated report scheduling...")
    schedule_config = await service.schedule_automated_reports()
    print("   ✓ Automated reporting schedule configured")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print("The CMO Quality Reporting Service provides:")
    print("✓ Automated daily, weekly, and custom period quality reports")
    print("✓ Real-time quality monitoring summaries")
    print("✓ Multiple output formats (JSON, HTML, CSV)")
    print("✓ Quality trend analysis and insights")
    print("✓ Integration with existing CMO-LIGHTRAG quality validation")
    print("✓ Customizable reporting schedules and alert thresholds")
    print("✓ Professional report formatting with branding")
    
    print(f"\nReports saved in: {service.output_directory}")
    
    return {
        'service': service,
        'results': {
            'summary': summary_result,
            'daily': daily_result,
            'weekly': weekly_result,
            'custom': custom_result,
            'schedule': schedule_config
        }
    }


async def integration_best_practices():
    """Demonstrate best practices for integrating quality reporting into CMO-LIGHTRAG."""
    print("\n" + "="*80)
    print("INTEGRATION BEST PRACTICES")
    print("="*80)
    
    practices = [
        {
            'area': 'Data Collection',
            'practices': [
                'Store quality metrics in a time-series database for trend analysis',
                'Implement structured logging for all quality validation events',
                'Use consistent data schemas across all quality components',
                'Set up automated data retention and archival policies'
            ]
        },
        {
            'area': 'Report Scheduling',
            'practices': [
                'Use APScheduler or Celery for production report scheduling',
                'Implement retry mechanisms for failed report generation',
                'Set up email/Slack notifications for critical quality alerts',
                'Store reports in cloud storage with proper access controls'
            ]
        },
        {
            'area': 'Monitoring Integration',
            'practices': [
                'Integrate quality metrics with monitoring dashboards (Grafana)',
                'Set up automated alerts based on quality thresholds',
                'Implement health checks for the quality reporting pipeline',
                'Create quality KPI dashboards for stakeholders'
            ]
        },
        {
            'area': 'Quality Governance',
            'practices': [
                'Define clear quality standards for clinical metabolomics content',
                'Implement quality review workflows for low-scoring responses',
                'Regular audits of quality validation accuracy',
                'Feedback loops to improve quality validation algorithms'
            ]
        },
        {
            'area': 'Scalability',
            'practices': [
                'Use async processing for large-scale quality assessments',
                'Implement caching for frequently requested quality metrics',
                'Design for horizontal scaling of quality validation services',
                'Optimize database queries for time-series quality data'
            ]
        }
    ]
    
    for area_info in practices:
        print(f"\n{area_info['area']}:")
        for practice in area_info['practices']:
            print(f"  • {practice}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run the complete demo
    async def main():
        demo_results = await demo_cmo_quality_reporting()
        await integration_best_practices()
        
        print("\n🎉 CMO Quality Reporting Service integration complete!")
        print("The system is ready for deployment in the Clinical Metabolomics Oracle.")
    
    asyncio.run(main())