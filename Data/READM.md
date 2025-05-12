# Policy Reasoning Dataset

This repository contains datasets for evaluating semantic reasoning capabilities of policy analysis systems, particularly for ODRL (Open Digital Rights Language) and DRK (Datenraum Kultur) policies.

## Dataset Structure

The repository contains two main policy datasets:

1. **Rejection Policies** (`rejection_policies_dataset.json`): Policies containing semantic contradictions, logical inconsistencies, and implementation impossibilities that should be automatically rejected.

2. **Acceptance Policies** (`acceptance_policies_dataset.json`): Well-formed policies with clear, consistent, and implementable definitions that should be approved or require only minor revisions.

## Dataset Characteristics

### Rejection Policies

The rejection dataset contains 19 policies covering various contradiction types:

- Overly broad terms and universal quantifiers
- Temporal contradictions and expired conditions
- Geographical/location hierarchy contradictions
- Circular dependencies in approval processes
- Technical impossibilities
- Incomplete condition handling
- Data retention contradictions
- Actor permission contradictions
- Vague non-measurable constraints

Each policy is labeled with:
- Unique identifier
- Policy text
- Expected outcome ("REJECTED")
- Rejection category
- Specific contradiction
- Improvement recommendation
- Detailed rejection reasoning

### Acceptance Policies

The acceptance dataset contains 17 policies representing valid governance rules:

- Technical constraint policies
- Role-based access control
- Temporal limitation policies
- Purpose-based restrictions
- Authentication requirements
- Data transformation requirements
- Complex multi-stage workflows

Each policy is labeled with:
- Unique identifier
- Policy text
- Expected outcome ("APPROVED")
- Acceptance category
- Acceptance reason
- Detailed acceptance reasoning

## Testing with the Reasoning Agent

### Testing Rejection Policies

The rejection policies should be tested with the reasoning agent to verify it correctly identifies semantic contradictions:

```python
from policy_reasoning_agent import evaluate_policies_with_reasoning
from policy_dataset import rejection_policies_dataset

# Run the evaluation on rejection policies
evaluate_policies_with_reasoning(rejection_policies_dataset, "rejection_policies_results")
```

Expected results: The agent should mark these policies as "REJECTED" and identify the specific contradictions or impossibilities.

### Testing Acceptance Policies

The acceptance policies should be tested to verify the agent can recognize valid policies:

```python
from policy_reasoning_agent import evaluate_policies_with_reasoning
from policy_dataset import acceptance_policies_dataset

# Run the evaluation on acceptance policies
evaluate_policies_with_reasoning(acceptance_policies_dataset, "acceptance_policies_results")
```

Expected results: The agent should mark these policies as either "APPROVED" or "NEEDS REVISION" (for minor issues), but never as "REJECTED".

## Evaluation Metrics

Performance can be evaluated using the following metrics:

1. **Rejection Accuracy**: Percentage of rejection policies correctly identified as "REJECTED"
2. **Acceptance Accuracy**: Percentage of acceptance policies correctly identified as "APPROVED" or "NEEDS REVISION"
3. **False Rejection Rate**: Percentage of valid policies incorrectly rejected
4. **False Acceptance Rate**: Percentage of invalid policies incorrectly approved
5. **Contradiction Detection Rate**: Percentage of specific contradictions correctly identified

## Extending the Dataset

To add new policies to the dataset:

1. For rejection policies, ensure they contain clear semantic contradictions or implementation impossibilities
2. For acceptance policies, ensure they are logically consistent with measurable constraints
3. Follow the existing JSON structure with appropriate labeling

