"""Command-line interface for AI Vehicle Monitoring System."""

import asyncio
import click
import os
from typing import Optional

from .data.ingestion import DataIngestionPipeline
from .core.vector_store import VectorStore
from .core.rag_pipeline import RAGPipeline
from .core.evaluation import RAGEvaluator
from .core.performance import performance_monitor
from .models.query import QueryRequest, QueryType, SearchMode
from .models.user import UserRole
from .utils.config import get_settings
from .utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@click.group()
def cli():
    """AI Vehicle Monitoring System CLI."""
    pass


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option('--vehicles', type=int, default=100, help='Number of sample vehicles to generate')
@click.option('--incidents', type=int, default=50, help='Number of sample incidents to generate')
def generate_samples(vehicles: int, incidents: int):
    """Generate sample data for testing."""
    click.echo(f"Generating {vehicles} sample vehicles and {incidents} sample incidents...")
    
    pipeline = DataIngestionPipeline()
    pipeline.generate_sample_data(vehicles, incidents)
    
    click.echo("Sample data generated successfully!")
    click.echo("Files created:")
    click.echo("  - data/samples/sample_vehicles.csv")
    click.echo("  - data/samples/sample_incidents.csv")
    click.echo("  - data/samples/sample_vehicles.json")
    click.echo("  - data/samples/sample_incidents.json")


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', 'data_type', type=click.Choice(['vehicles', 'incidents']), required=True)
def ingest_csv(file_path: str, data_type: str):
    """Ingest data from CSV file."""
    click.echo(f"Ingesting {data_type} from {file_path}...")
    
    async def run_ingestion():
        pipeline = DataIngestionPipeline()
        
        if data_type == 'vehicles':
            stats = await pipeline.ingest_vehicle_csv(file_path)
        else:
            stats = await pipeline.ingest_incident_csv(file_path)
        
        return stats
    
    stats = asyncio.run(run_ingestion())
    
    click.echo("Ingestion completed!")
    click.echo(f"Processed: {stats['processed']}")
    click.echo(f"Errors: {stats['errors']}")
    click.echo(f"Skipped: {stats['skipped']}")


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', 'data_type', type=click.Choice(['vehicles', 'incidents']), required=True)
def ingest_json(file_path: str, data_type: str):
    """Ingest data from JSON file."""
    click.echo(f"Ingesting {data_type} from {file_path}...")
    
    async def run_ingestion():
        pipeline = DataIngestionPipeline()
        stats = await pipeline.ingest_json_data(file_path, data_type)
        return stats
    
    stats = asyncio.run(run_ingestion())
    
    click.echo("Ingestion completed!")
    click.echo(f"Processed: {stats['processed']}")
    click.echo(f"Errors: {stats['errors']}")
    click.echo(f"Skipped: {stats['skipped']}")


@cli.group()
def vector():
    """Vector store management commands."""
    pass


@vector.command()
def stats():
    """Show vector store statistics."""
    vector_store = VectorStore()
    stats = vector_store.get_stats()
    
    click.echo("Vector Store Statistics:")
    click.echo(f"  Total documents: {stats['total_documents']}")
    click.echo(f"  Vehicles: {stats['vehicle_count']}")
    click.echo(f"  Incidents: {stats['incident_count']}")
    click.echo(f"  Dimension: {stats['dimension']}")
    click.echo(f"  Index trained: {stats['index_trained']}")


@vector.command()
def rebuild():
    """Rebuild the vector index."""
    click.echo("Rebuilding vector index...")
    
    vector_store = VectorStore()
    vector_store.rebuild_index()
    
    click.echo("Vector index rebuilt successfully!")


@cli.group()
def query():
    """Query system commands."""
    pass


@query.command()
@click.argument('query_text')
@click.option('--role', type=click.Choice(['officer', 'analyst', 'administrator', 'supervisor']), 
              default='officer', help='User role for context shaping')
@click.option('--type', 'query_type', type=click.Choice(['vehicle_search', 'incident_search', 'risk_assessment', 'pattern_analysis', 'general_inquiry']), 
              default='general_inquiry', help='Query type')
@click.option('--mode', type=click.Choice(['semantic', 'exact', 'hybrid']), 
              default='hybrid', help='Search mode')
@click.option('--max-results', type=int, default=10, help='Maximum number of results')
def search(query_text: str, role: str, query_type: str, mode: str, max_results: int):
    """Execute a search query."""
    click.echo(f"Executing query: {query_text}")
    click.echo(f"Role: {role}, Type: {query_type}, Mode: {mode}")
    
    async def run_query():
        # Create query request
        query_request = QueryRequest(
            query_text=query_text,
            query_type=QueryType(query_type),
            search_mode=SearchMode(mode),
            user_role=UserRole(role),
            max_results=max_results
        )
        
        # Execute query
        rag_pipeline = RAGPipeline()
        response = await rag_pipeline.process_query(query_request)
        
        return response
    
    try:
        response = asyncio.run(run_query())
        
        click.echo("\n" + "="*80)
        click.echo("QUERY RESPONSE")
        click.echo("="*80)
        click.echo(f"Query ID: {response.query_id}")
        click.echo(f"Confidence Score: {response.confidence_score:.2f}")
        click.echo(f"Processing Time: {response.processing_time_ms}ms")
        click.echo(f"Vehicle Results: {len(response.vehicle_results)}")
        click.echo(f"Incident Results: {len(response.incident_results)}")
        click.echo("\nResponse:")
        click.echo("-" * 40)
        click.echo(response.response_text)
        
        if response.vehicle_results:
            click.echo("\nTop Vehicle Results:")
            click.echo("-" * 40)
            for i, result in enumerate(response.vehicle_results[:3], 1):
                vehicle = result.vehicle
                click.echo(f"{i}. {vehicle.registration_number} (Risk: {vehicle.risk_score:.2f}, Score: {result.similarity_score:.2f})")
        
        if response.incident_results:
            click.echo("\nTop Incident Results:")
            click.echo("-" * 40)
            for i, result in enumerate(response.incident_results[:3], 1):
                incident = result.incident
                click.echo(f"{i}. {incident.incident_id} ({incident.incident_type}, Severity: {incident.severity_level}, Score: {result.relevance_score:.2f})")
        
    except Exception as e:
        click.echo(f"Query failed: {str(e)}", err=True)


@cli.group()
def eval():
    """Evaluation and benchmarking commands."""
    pass


@cli.group()
def perf():
    """Performance monitoring commands."""
    pass


@cli.group()
def system():
    """System management commands."""
    pass


@system.command()
def status():
    """Show system status."""
    click.echo("AI Vehicle Monitoring System Status")
    click.echo("=" * 40)
    
    # Check vector store
    try:
        vector_store = VectorStore()
        vector_stats = vector_store.get_stats()
        click.echo(f"✓ Vector Store: {vector_stats['total_documents']} documents")
    except Exception as e:
        click.echo(f"✗ Vector Store: Error - {str(e)}")
    
    # Check data directories
    data_dirs = [
        settings.faiss_index_path,
        settings.chromadb_path,
        "data/samples",
        "logs"
    ]
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            click.echo(f"✓ Directory: {dir_path}")
        else:
            click.echo(f"✗ Directory: {dir_path} (missing)")
    
    # Check configuration
    click.echo(f"✓ OpenAI Model: {settings.openai_model}")
    click.echo(f"✓ Embedding Model: {settings.embedding_model}")
    click.echo(f"✓ Vector Dimension: {settings.vector_dimension}")


@system.command()
def setup():
    """Setup the system for first use."""
    click.echo("Setting up AI Vehicle Monitoring System...")
    
    # Create directories
    directories = [
        settings.faiss_index_path,
        settings.chromadb_path,
        "data/samples",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        click.echo(f"✓ Created directory: {directory}")
    
    # Initialize vector store
    try:
        vector_store = VectorStore()
        click.echo("✓ Initialized vector store")
    except Exception as e:
        click.echo(f"✗ Failed to initialize vector store: {str(e)}")
    
    # Generate sample data
    if click.confirm("Generate sample data for testing?"):
        pipeline = DataIngestionPipeline()
        pipeline.generate_sample_data(50, 25)
        click.echo("✓ Generated sample data")
        
        # Ingest sample data
        if click.confirm("Ingest sample data into vector store?"):
            async def ingest_samples():
                stats1 = await pipeline.ingest_vehicle_csv("data/samples/sample_vehicles.csv")
                stats2 = await pipeline.ingest_incident_csv("data/samples/sample_incidents.csv")
                return stats1, stats2
            
            try:
                vehicle_stats, incident_stats = asyncio.run(ingest_samples())
                click.echo(f"✓ Ingested {vehicle_stats['processed']} vehicles")
                click.echo(f"✓ Ingested {incident_stats['processed']} incidents")
            except Exception as e:
                click.echo(f"✗ Failed to ingest sample data: {str(e)}")
    
    click.echo("\nSystem setup completed!")
    click.echo("You can now:")
    click.echo("  - Run the API server: uvicorn src.api.main:app --reload")
    click.echo("  - Test queries: python -m src.cli query search 'your query here'")
    click.echo("  - Check status: python -m src.cli system status")


@eval.command()
@click.option('--ground-truth', type=click.Path(exists=True), help='Path to ground truth JSON file')
@click.option('--output', type=click.Path(), help='Path to save results')
def benchmark(ground_truth: str, output: str):
    """Run benchmark evaluation."""
    if not ground_truth:
        click.echo("Error: Ground truth file required", err=True)
        return

    async def run_benchmark():
        evaluator = RAGEvaluator()
        rag_pipeline = RAGPipeline()
        evaluator.rag_pipeline = rag_pipeline

        # Load ground truth
        evaluator.load_ground_truth(ground_truth)

        # Run benchmark
        click.echo("Running benchmark evaluation...")
        results = await evaluator.run_benchmark()

        # Display results
        click.echo("\n" + "="*60)
        click.echo("BENCHMARK RESULTS")
        click.echo("="*60)
        click.echo(f"Total Queries: {results.total_queries}")
        click.echo(f"Precision@5: {results.overall_precision_at_5:.3f}")
        click.echo(f"Precision@10: {results.overall_precision_at_10:.3f}")
        click.echo(f"Recall@5: {results.overall_recall_at_5:.3f}")
        click.echo(f"Recall@10: {results.overall_recall_at_10:.3f}")
        click.echo(f"Average BLEU Score: {results.average_bleu_score:.3f}")
        click.echo(f"Average Response Time: {results.average_response_time:.1f}ms")
        click.echo(f"Average Confidence: {results.average_confidence:.3f}")
        click.echo(f"False Positive Rate: {results.false_positive_rate:.3f}")

        # Save results if output path provided
        if output:
            evaluator.save_results(results, output)
            click.echo(f"\nResults saved to: {output}")

        return results

    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        click.echo(f"Benchmark failed: {str(e)}", err=True)


@eval.command()
@click.argument('output_path', type=click.Path())
def generate_ground_truth(output_path: str):
    """Generate sample ground truth data."""
    evaluator = RAGEvaluator()
    evaluator.generate_sample_ground_truth(output_path)
    click.echo(f"Sample ground truth generated: {output_path}")


@perf.command()
def status():
    """Show current performance metrics."""
    metrics = performance_monitor.get_current_metrics()

    click.echo("Performance Metrics")
    click.echo("=" * 40)
    click.echo(f"Uptime: {metrics.get('uptime_seconds', 0):.0f} seconds")
    click.echo(f"Query Rate: {metrics.get('query_rate', 0):.2f} queries/sec")
    click.echo(f"Avg Response Time: {metrics.get('avg_response_time', 0):.1f}ms")
    click.echo(f"Error Rate: {metrics.get('error_rate', 0):.3f}")
    click.echo(f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.3f}")
    click.echo(f"CPU Usage: {metrics.get('cpu_percent', 0):.1f}%")
    click.echo(f"Memory Usage: {metrics.get('memory_percent', 0):.1f}%")
    click.echo(f"Memory Used: {metrics.get('memory_used_mb', 0):.1f}MB")


@perf.command()
def summary():
    """Show comprehensive performance summary."""
    summary = performance_monitor.get_performance_summary()

    click.echo("Performance Summary")
    click.echo("=" * 50)

    # Current metrics
    current = summary['current_metrics']
    click.echo("\nCurrent Metrics:")
    for key, value in current.items():
        if isinstance(value, float):
            click.echo(f"  {key}: {value:.2f}")
        else:
            click.echo(f"  {key}: {value}")

    # Target compliance
    compliance = summary['target_compliance']
    click.echo("\nTarget Compliance:")
    for target, compliant in compliance.items():
        status = "✓" if compliant else "✗"
        click.echo(f"  {status} {target}")

    # Overall health
    health = "HEALTHY" if summary['overall_health'] else "DEGRADED"
    click.echo(f"\nOverall Health: {health}")


if __name__ == '__main__':
    cli()
