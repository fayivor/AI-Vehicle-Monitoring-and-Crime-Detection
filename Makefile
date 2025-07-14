# AI Vehicle Monitoring & Crime Detection System - Makefile
# Government of Ghana - Ministry of Local Government and Rural Development
# CLASSIFICATION: RESTRICTED - FOR OFFICIAL USE ONLY

.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-run setup-dev

# Default target
help:
	@echo "AI Vehicle Monitoring & Crime Detection System - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup-dev    Setup development environment"
	@echo "  run          Run the development server"
	@echo ""
	@echo "Code Quality:"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean up temporary files"

# Installation
install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/dev.txt

setup-dev: install-dev
	@echo "Setting up secure development environment..."
	@echo "SECURITY NOTICE: Ensure you have appropriate clearance before proceeding."
	mkdir -p data/faiss_index data/chromadb logs
	cp .env.example .env
	@echo "Development environment setup complete."
	@echo "IMPORTANT: Edit .env file with classified configuration parameters."
	@echo "Contact IT Security Division for production credentials."

# Development server
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

# Docker
docker-build:
	docker build -t ai-vehicle-system .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(message)"

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/

# Security
security-check:
	safety check
	bandit -r src/

# Documentation
docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

# Production deployment
deploy-staging:
	@echo "Deploying to staging environment..."
	# Add staging deployment commands here

deploy-prod:
	@echo "Deploying to production environment..."
	# Add production deployment commands here

# Data management
data-backup:
	@echo "Backing up data..."
	# Add data backup commands here

data-restore:
	@echo "Restoring data..."
	# Add data restore commands here

# Performance testing
perf-test:
	@echo "Running performance tests..."
	# Add performance testing commands here

# All quality checks
check-all: lint type-check test security-check
	@echo "All quality checks passed!"
