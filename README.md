[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-brightgreen.svg)](https://www.langchain.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![Issues](https://img.shields.io/github/issues/Daham-Mustaf/ODRL_Policy-Reasoning-Agents)


# ODRL Policy-Reasoning Agents

An ontology-grounded multi-agent system for reliable Open Digital Rights Language (ODRL) policy generation and validation.

## Overview

This framework addresses the challenges of translating natural language policy descriptions into enforceable ODRL representations for data spaces. It employs a three-agent architecture to overcome limitations of direct LLM-based policy generation:

1. **Policy Reasoning Agent** - Detects semantic contradictions and logical conflicts in natural language policy descriptions
2. **ODRL Generation Agent** - Transforms validated policy structures into ontology-aligned ODRL representations
3. **ODRL Validation Agent** - Applies SHACL validation to ensure syntactic correctness and semantic soundness

## Key Features

- **Semantic Conflict Detection** - Identifies and resolves six types of policy conflicts:
  - Vague/overbroad policies (e.g., "everyone can access everything")
  - Spatial conflicts (e.g., permission in Germany, prohibition in EU)
  - Temporal conflicts (e.g., overlapping time windows with contradictory rules)
  - Action conflicts (e.g., permitting "share" while prohibiting "distribute")
  - Dependency conflicts (e.g., circular approval workflows)
  - Role/context conflicts (e.g., contradictory permissions across hierarchical roles)

- **Ontology-Grounded Generation** - Ensures ODRL vocabulary compliance and prevents hallucination
- **SHACL Validation** - Enforces policy compliance through 10 specialized node shapes with 62 property validations

## Repository Structure

```
.
├── policy_reasoning_agent.py       # Agent 1: Semantic reasoning and conflict detection
├── policy_reasoning_utils.py       # Utility functions for reasoning agent
├── odrl_policy_generation_agent.py # Agent 2: ODRL generation from validated structures
├── odrl_policy_validator_agent.py  # Agent 3: SHACL validation and correction
├── utils/                          # Shared utility functions
│   └── data_utils.py               # Data handling utilities
├── odrl_shacl_shapes/              # SHACL validation rules
│   └── shacl_shapes.ttl            # SHACL shapes for ODRL validation
├── Data/                           # Policy datasets
│   ├── acceptance_policies/        # Valid implementable policies
│   └── rejection_policies/         # Policies with logical conflicts
├── agent_reasoning_results/        # Reasoning results by model
│   ├── gpt-4o/                     # GPT-4o reasoning outputs 
│   ├── gpt-4o-mini/                # GPT-4o-mini reasoning outputs
│   ├── gpt-35-turbo/               # GPT-3.5 Turbo reasoning outputs
│   └── o3-mini/                    # Claude Opus Mini reasoning outputs
└── agent_generated_odrl/           # Generated ODRL policies by model
    ├── gpt-4o/                     # GPT-4o ODRL outputs
    ├── gpt-4o-mini/                # GPT-4o-mini ODRL outputs
    ├── gpt-35-turbo/               # GPT-3.5 Turbo ODRL outputs
    └── o3-mini/                    # Claude Opus Mini ODRL outputs
```

## Usage

### Prerequisites

- Python 3.8+
- Required packages: transformers, rdflib, pyshacl, langgraph, openai

### Basic Usage

1. Run the policy reasoning agent to detect conflicts:

```python
python policy_reasoning_agent.py --input-file your_policy.txt --model gpt-4o
```

2. Generate ODRL for validated policies:

```python
python odrl_policy_generation_agent.py --input-file validated_policy.json --model gpt-4o --output-format ttl
```

3. Validate generated ODRL:

```python
python odrl_policy_validator_agent.py --input-file generated_policy.ttl
```

## Evaluation Results

The framework has been evaluated on 37 policy statements from the cultural data domain:
- 20 logically inconsistent policies with various conflict types
- 17 valid implementable policies

Performance metrics across models (see paper for details):
- GPT-4o: 97.3% overall accuracy (19/20 conflicts detected, 17/17 valid policies approved)
- GPT-4o-Mini: 94.6% accuracy
- GPT-3.5-Turbo: 86.5% accuracy
- Claude O3-Mini: 94.6% accuracy

