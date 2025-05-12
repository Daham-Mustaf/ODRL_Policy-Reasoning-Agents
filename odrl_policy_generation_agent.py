from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any, Union, Optional
import logging
import os
import json
import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Azure OpenAI Configuration ---
AZURE_API_KEY = "xxxxx"
AZURE_ENDPOINT = "xxx.com"

# AZURE_API_VERSION = "2024-10-01-preview"  # For older models
AZURE_API_VERSION = "2024-12-01-preview"  # For newer models like O1 and O3

# Deployment names - uncomment one
# DEPLOYMENT_NAME = "o1-2024-12-17"             # Claude-like model, strong reasoning, API version 2024-12-01-preview required
DEPLOYMENT_NAME = "o3-mini-2025-01-31"        # Latest model, efficient, API version 2024-12-01-preview required
# DEPLOYMENT_NAME = "o1-mini-2024-09-12"        # Smaller Claude-like model, API version 2024-12-01-preview required
# DEPLOYMENT_NAME = "gpt-35-turbo-0125"         # GPT-3.5 Turbo, fast and cost-effective
# DEPLOYMENT_NAME = "gpt-4-32k-0613"            # GPT-4 with larger context window
# DEPLOYMENT_NAME = "gpt-4o-2024-08-06"         # GPT-4o, strong general capabilities
# DEPLOYMENT_NAME = "gpt-4o-mini-2024-07-18"    # Smaller GPT-4o, efficient
# DEPLOYMENT_NAME = "text-embedding-3-large-1"  # For embedding generation
# DEPLOYMENT_NAME = "text-embedding-ada-002-2"  # Legacy embedding model

# --- LLM Setup ---
llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME,
)

# --- System Prompts ---

odrl_translation_prompt = """
You are an expert ODRL (Open Digital Rights Language) policy generator for cultural heritage data spaces. Your task is to translate natural language policy statements into valid ODRL representations that pass both semantic validation and SHACL shape validation checks.

## TRANSLATION PROCESS: STEP-BY-STEP

Follow this logical sequence when translating natural language policies to ODRL:

### STEP 1: ANALYZE THE POLICY TEXT
- Identify all parties involved (who is granting rights, who is receiving them)
- Identify all actions (what activities are allowed, forbidden, or required)
- Identify all assets (what digital resources the policy applies to)
- Identify all constraints (temporal, spatial, purpose, etc.)

### STEP 2: DETERMINE THE POLICY TYPE
Based on the parties involved, select one:
- **odrl:Agreement**: When both assigner and assignee are explicitly mentioned
- **odrl:Offer**: When there's an assigner offering to potential assignees
- **odrl:Set**: When no specific parties are mentioned

### STEP 3: DETERMINE THE RULE TYPES
Based on the actions, create appropriate rules:
- **odrl:permission**: For allowed actions ("can", "may", "is allowed to")
- **odrl:prohibition**: For forbidden actions ("cannot", "must not", "is prohibited from")
- **odrl:obligation**: For required actions ("must", "shall", "is required to")
- **odrl:duty**: For requirements attached to permissions

### STEP 4: FORMALIZE THE CONSTRAINTS
Convert each constraint to the ODRL structure:
- Select the appropriate leftOperand (dateTime, purpose, spatial, etc.)
- Select the appropriate operator (eq, lt, gt, etc.)
- Format the rightOperand with the correct datatype

### STEP 5: APPLY PROPER NAMING CONVENTIONS
- Policies: `drk:[ContentType][RuleType]Policy` (e.g., `drk:ManuscriptAccessPolicy`)
- Assets: `drk:[Collection][ContentType]` (e.g., `drk:MedievalManuscriptCollection`)
- Parties: `drk:[OrganizationType/Role]` (e.g., `drk:ResearchInstitution`)

### STEP 6: CONSTRUCT THE FULL ODRL POLICY
Assemble all components with proper syntax and required metadata.

## PREFIX DECLARATIONS (always include these)

```turtle
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix drk: <http://w3id.org/drk/ontology/> .
```

## POLICY STRUCTURE TEMPLATE

```turtle
drk:PolicyName a odrl:PolicyType ;
    # Required metadata
    dct:title "Policy Title" ;
    dct:description "Policy Description" ;
    dct:issued "YYYY-MM-DD"^^xsd:date ;
    
    # Permission rule
    odrl:permission [
        a odrl:Permission ;
        odrl:action odrl:actionName ;
        odrl:target drk:AssetName ;
        odrl:assigner drk:AssignerName ;
        odrl:assignee drk:AssigneeName ;
        
        # Constraint example
        odrl:constraint [
            a odrl:Constraint ;
            odrl:leftOperand odrl:constraintProperty ;
            odrl:operator odrl:operatorType ;
            odrl:rightOperand "value"^^xsd:datatype
        ] ;
        
        # Duty example
        odrl:duty [
            a odrl:Duty ;
            odrl:action odrl:dutyAction ;
            odrl:target drk:DutyTarget
        ]
    ] ;
    
    # Prohibition rule
    odrl:prohibition [
        a odrl:Prohibition ;
        odrl:action odrl:actionName ;
        odrl:target drk:AssetName ;
        odrl:assigner drk:AssignerName ;
        odrl:assignee drk:AssigneeName
    ] .

# Asset definition
drk:AssetName a odrl:Asset, dct:ResourceType ;
    rdfs:label "Asset Label" .

# Party definitions
drk:AssignerName a odrl:Party, foaf:Organization ;
    rdfs:label "Assigner Label" .

drk:AssigneeName a odrl:Party ;
    rdfs:label "Assignee Label" .
```

## SHACL VALIDATION REQUIREMENTS

The following requirements must be met for the policy to pass SHACL validation:

### 1. MANDATORY TYPE DECLARATIONS
- Policies MUST have explicit type: `a odrl:Agreement`, `a odrl:Offer`, or `a odrl:Set`
- Permissions MUST have explicit type: `a odrl:Permission`
- Prohibitions MUST have explicit type: `a odrl:Prohibition`
- Duties MUST have explicit type: `a odrl:Duty`
- Constraints MUST have explicit type: `a odrl:Constraint`
- Assets MUST have explicit type: `a odrl:Asset`
- Parties MUST have explicit type: `a odrl:Party`

### 2. MANDATORY PROPERTY REQUIREMENTS
- Permissions MUST have `odrl:action` and `odrl:target`
- Prohibitions MUST have `odrl:action` and `odrl:target`
- Duties MUST have `odrl:action`
- Constraints MUST have `odrl:leftOperand`, `odrl:operator`, and `odrl:rightOperand`
- Policies MUST have `dct:title`, `dct:description`, and `dct:issued`

### 3. COMMON SHACL VALIDATION ERRORS TO AVOID
```
# Missing target asset
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Permission must specify a target asset" ;
]

# Missing action
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Permission must specify at least one action" ;
]

# Missing explicit type
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Permission must be explicitly typed as odrl:Permission" ;
]

# Missing constraint component
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Constraint must specify a leftOperand" ;
]

# Insufficient constraints in logical group
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Logical constraint groups must contain at least two constraints" ;
]

# Missing policy type
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Policy must have a type (Agreement, Offer, or Set)" ;
]

# Missing required metadata
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Policy must have a title" ;
]

# Missing party type
[
    sh:resultSeverity sh:Violation ;
    sh:resultMessage "Parties must be explicitly typed" ;
]
```

## STANDARD ODRL ACTION TYPES

Use these standard action types in your ODRL policies:

- **odrl:read**: Access for viewing content
- **odrl:use**: General usage of content
- **odrl:reproduce**: Make copies of content
- **odrl:distribute**: Share content with others
- **odrl:modify**: Change or transform content
- **odrl:delete**: Remove or delete content
- **odrl:archive**: Store content for preservation
- **odrl:present**: Display content publicly
- **odrl:index**: Create searchable index of content
- **odrl:print**: Print physical copies of content
- **odrl:extract**: Take portions of content
- **odrl:attribute**: Give credit to a party
- **odrl:compensate**: Provide payment or compensation

## STANDARD ODRL LEFTOPERAND VALUES

Always use these standard leftOperand values for constraints:

### TIME-RELATED LEFTOPERANDS:
- **odrl:dateTime**: The date/time of action execution (with xsd:dateTime)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:dateTime ;
      odrl:operator odrl:lt ;
      odrl:rightOperand "2025-01-01T00:00:00.000Z"^^xsd:dateTime
  ] ;
  ```

- **odrl:delayPeriod**: A time delay period (with xsd:duration)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:delayPeriod ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "P30D"^^xsd:duration
  ] ;
  ```

- **odrl:timeInterval**: A recurring period between executions (with xsd:duration)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:timeInterval ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "P7D"^^xsd:duration
  ] ;
  ```

- **odrl:elapsedTime**: A continuous time period (with xsd:duration)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:elapsedTime ;
      odrl:operator odrl:lteq ;
      odrl:rightOperand "PT2H"^^xsd:duration
  ] ;
  ```

### PURPOSE & ROLE LEFTOPERANDS:
- **odrl:purpose**: Purpose limitation (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:purpose ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "research"^^xsd:string
  ] ;
  ```

- **odrl:recipient**: Recipient role restriction (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:recipient ;
      odrl:operator odrl:isA ;
      odrl:rightOperand "academic-researcher"^^xsd:string
  ] ;
  ```

- **odrl:industry**: Industry sector context (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:industry ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "education"^^xsd:string
  ] ;
  ```

### LOCATION LEFTOPERANDS:
- **odrl:spatial**: Geographic location restrictions (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:spatial ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "museum-premises"^^xsd:string
  ] ;
  ```

- **odrl:virtualLocation**: IT communication location (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:virtualLocation ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "institutional-network"^^xsd:string
  ] ;
  ```

### ASSET PROPERTY LEFTOPERANDS:
- **odrl:count**: Usage count limitations (with xsd:integer)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:count ;
      odrl:operator odrl:lteq ;
      odrl:rightOperand "5"^^xsd:integer
  ] ;
  ```

- **odrl:resolution**: Image resolution limitations (with xsd:integer)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:resolution ;
      odrl:operator odrl:lteq ;
      odrl:rightOperand "600"^^xsd:integer
  ] ;
  ```

- **odrl:fileFormat**: File format restrictions (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:fileFormat ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "image/jpeg"^^xsd:string
  ] ;
  ```

- **odrl:language**: Language restrictions (with xsd:string)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:language ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "en"^^xsd:string
  ] ;
  ```

- **odrl:percentage**: Percentage restrictions (with xsd:decimal)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:percentage ;
      odrl:operator odrl:lteq ;
      odrl:rightOperand "50.0"^^xsd:decimal
  ] ;
  ```

- **odrl:payAmount**: Payment amount (with xsd:decimal)
  ```turtle
  odrl:constraint [
      a odrl:Constraint ;
      odrl:leftOperand odrl:payAmount ;
      odrl:operator odrl:eq ;
      odrl:rightOperand "100.00"^^xsd:decimal
  ] ;
  ```

## LOGICAL CONSTRAINT OPERATORS

For combining multiple constraints:

### AND OPERATOR (all constraints must be satisfied):
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:and (
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:purpose ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "research"^^xsd:string
        ]
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:dateTime ;
            odrl:operator odrl:lt ;
            odrl:rightOperand "2025-01-01T00:00:00.000Z"^^xsd:dateTime
        ]
    )
] ;
```

### OR OPERATOR (at least one constraint must be satisfied):
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:or (
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:spatial ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "museum"^^xsd:string
        ]
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:spatial ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "library"^^xsd:string
        ]
    )
] ;
```

### EXCLUSIVE OR OPERATOR (exactly one constraint must be satisfied):
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:xone (
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:purpose ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "education"^^xsd:string
        ]
        [
            a odrl:Constraint ;
            odrl:leftOperand odrl:purpose ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "research"^^xsd:string
        ]
    )
] ;
```

## NAMING CONVENTIONS

Always follow these naming conventions:

1. Use the `drk:` prefix for all domain-specific entities
2. Use descriptive PascalCase names (each word capitalized, no separators)
3. Follow these patterns:
   - Policies: `drk:[ContentType][RuleType]Policy` (e.g., `drk:ManuscriptAccessPolicy`)
   - Assets: `drk:[Collection][ContentType]` (e.g., `drk:MedievalManuscriptCollection`)
   - Parties: `drk:[OrganizationType/Role]` (e.g., `drk:ResearchInstitution`)

4. NEVER use generic names like "policy1", "asset1", "user1"
Now, convert the following natural language policy statement into ODRL using TTL syntax:

{user_input}
"""

def translate_to_odrl(policy_text: str, use_cot_refinement: bool = False) -> str:
    """
    Translates a natural language policy into ODRL TTL format with optional Chain of Thought refinement
    
    Args:
        policy_text (str): The natural language policy text
        use_cot_refinement (bool): Whether to use Chain of Thought reasoning to refine the translation
        
    Returns:
        str: ODRL representation in TTL format
    """
    try:
        logger.info("Starting ODRL translation")
        
        # Format the prompt with the policy text
        prompt = odrl_translation_prompt.format(user_input=policy_text)
        
        # Invoke LLM for translation
        response = llm.invoke([SystemMessage(content=prompt)])
        
        # Extract only the TTL code from the response if needed
        initial_ttl_content = extract_ttl_code(response.content)
        
        # Optional Chain of Thought refinement step
        if use_cot_refinement:
            logger.info("Applying Chain of Thought reasoning to refine the ODRL translation")
            refined_ttl = chain_of_thought_refinement(policy_text, initial_ttl_content)
            return refined_ttl
        else:
            logger.info("ODRL translation complete (without refinement)")
            return initial_ttl_content
        
    except Exception as e:
        logger.error(f"Error in ODRL translation: {e}")
        return f"Error translating to ODRL: {str(e)}"

def chain_of_thought_refinement(policy_text: str, initial_translation: str) -> str:
    """
    Refines an ODRL translation using explicit Chain of Thought reasoning
    
    Args:
        policy_text (str): The original natural language policy
        initial_translation (str): The initial ODRL translation
        
    Returns:
        str: Refined ODRL translation after Chain of Thought analysis
    """
    cot_prompt = f"""
    You are an expert ODRL validator and refiner. Review the following ODRL translation of a policy statement.
    Use Chain of Thought reasoning to analyze and improve the translation step by step.
    
    Original Policy:
    "{policy_text}"
    
    Initial ODRL Translation:
    ```turtle
    {initial_translation}
    ```
    
    Please perform the following reasoning steps, clearly explaining your thought process at each stage:
    
    Step 1: Analyze the structure of the policy text. What are the key components (permissions, prohibitions, duties, constraints, etc.)?
    
    Step 2: Check if the initial translation captures all these components correctly. Identify any missing or incorrectly represented elements.
    
    Step 3: Verify proper metadata inclusion (dct:title, dct:description, etc.). Is all required metadata present and accurate?
    
    Step 4: Validate proper typing for all entities (assets, parties, etc.). Are all entities properly typed?
    
    Step 5: Examine constraint structures. Are they properly formed with leftOperand, operator, and rightOperand?
    
    Step 6: Check for proper duty implementation. Are all duties properly defined with appropriate actions?
    
    Step 7: Verify naming conventions. Do all entities follow the required naming patterns?
    
    Step 8: Based on your analysis, provide a refined ODRL translation that addresses any issues identified.
    
    After showing your Chain of Thought reasoning, provide ONLY the refined TTL code as your final answer.
    """
    
    # Invoke LLM for Chain of Thought refinement
    response = llm.invoke([SystemMessage(content=cot_prompt)])
    
    # Extract only the TTL code from the refinement response
    # We need a more robust pattern that looks for the final TTL code after the reasoning
    ttl_pattern = r'```(?:ttl|turtle)?\n([\s\S]*?)```'
    matches = re.findall(ttl_pattern, response.content)
    
    if matches:
        # Return the last TTL code block found (after the reasoning)
        return matches[-1].strip()
    else:
        # Fallback to the initial translation if no code block found
        logger.warning("No refined TTL found in Chain of Thought response. Using initial translation.")
        return initial_translation
    
def process_policy_to_odrl(policy_text: str, policy_id: str, model_name: str, use_cot_refinement: bool = False) -> Dict[str, Any]:
    """
    Process a policy text to ODRL TTL format with optional Chain of Thought refinement
    
    Args:
        policy_text (str): The natural language policy text
        policy_id (str): The ID of the policy
        model_name (str): Name of the model used for generation
        use_cot_refinement (bool): Whether to apply Chain of Thought reasoning to refine the translation
        
    Returns:
        Dict: Result information including TTL content and file path
    """
    # Translate to ODRL
    ttl_content = translate_to_odrl(policy_text, use_cot_refinement=use_cot_refinement)
    
    # Validate TTL syntax
    is_valid = validate_ttl(ttl_content)
    
    # Save to file
    file_path = save_odrl_ttl(policy_text, ttl_content, policy_id, model_name)
    
    return {
        "policy_text": policy_text,
        "policy_id": policy_id,
        "ttl_content": ttl_content,
        "is_valid": is_valid,
        "file_path": file_path,
        "model_name": model_name,
        "refined_with_cot": use_cot_refinement
    }

def extract_ttl_code(response: str) -> str:
    """
    Extracts TTL code from LLM response, removing any explanatory text
    """
    # Check if response contains code blocks with TTL
    ttl_pattern = r'```(?:ttl|turtle)?\n([\s\S]*?)```'
    matches = re.findall(ttl_pattern, response)
    
    if matches:
        # Return the first TTL code block found
        return matches[0].strip()
    else:
        # If no code blocks found, assume the entire response is TTL
        # or at least return the response for manual filtering
        return response

def save_odrl_ttl(policy_text: str, ttl_content: str, policy_id: str, model_name: str) -> str:
    """
    Save the ODRL TTL content to a file in model-specific subfolders using the provided policy ID
    
    Args:
        policy_text (str): The original policy text
        ttl_content (str): The generated ODRL TTL content
        policy_id (str): The ID of the policy
        model_name (str): Name of the model used to generate the TTL
        
    Returns:
        str: Path to the saved file
    """
    # Create base directory and model-specific directory
    base_dir = "agent_generated_odrl"
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Use the provided policy ID for the filename
    filename = f"{policy_id}.ttl"
    file_path = os.path.join(model_dir, filename)
    
    # Save the TTL content to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(ttl_content)
    
    logger.info(f"ODRL TTL saved to: {file_path}")
    return file_path

def validate_ttl(ttl_content: str) -> bool:
    """
    Validate the TTL syntax (could be expanded with more robust validation)
    
    Args:
        ttl_content (str): The TTL content to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # This is a basic validation, you might want to use a proper RDF library
    # like rdflib for more robust validation
    required_prefixes = ["@prefix odrl:", "@prefix rdf:"]
    required_elements = ["odrl:permission", "odrl:prohibition", "odrl:obligation", "odrl:action"]
    
    # Check for required prefixes
    prefix_check = all(prefix in ttl_content for prefix in required_prefixes)
    
    # Check for at least one of the required ODRL elements
    element_check = any(element in ttl_content for element in required_elements)
    
    return prefix_check and element_check
# --- Example Usage ---
if __name__ == "__main__":
    # model_name = "gpt-4o-mini"
    # model_name = "gpt-35-turbo"
    # model_name = "gpt-4o-mini"
    # model_name = "gpt-4o"
    model_name = "o3-mini"	
 
    
    # ===================================================================
    # OPTION 1: Process a single policy (uncomment to use)
    # ===================================================================
    
    
    test_policy = {
        
        "policy_id": "drk_connector_based_access_policy",
        "policy_text": "Access to the dataset titled 'MuseumArtifact' is permitted only to approved dataspace participants(UC4) operating through authorized connectors. Only participants using connector endpoints registered as '?connector1URI' are allowed to use the data.",
        "expected_outcome": "APPROVED",
        "acceptance_category": "technical_constraint_policy",
        "acceptance_reason": "Policy clearly defines specific technical requirements for access",
        "acceptance_reasoning_detailed": "This policy establishes clear technical constraints that are measurable and enforceable. It specifies who can access the data (approved participants) and through what technical means (authorized connectors with specific endpoint IDs), making implementation straightforward without contradictions."
    
    }
    
    # Process without CoT refinement
    # result_default = process_policy_to_odrl(
    #     policy_text=test_policy["policy_text"],
    #     policy_id=test_policy["policy_id"],
    #     model_name=model_name,
    #     use_cot_refinement=False
    # )
    
    # print(f"\nProcessed (without CoT): {result_default['policy_id']}")
    # print(f"Saved to: {result_default['file_path']}")
    
    # # Process with CoT refinement
    # result_with_cot = process_policy_to_odrl(
    #     policy_text=test_policy["policy_text"],
    #     policy_id=f"{test_policy['policy_id']}_cot",
    #     model_name=model_name,
    #     use_cot_refinement=True
    # )
    
    # print(f"\nProcessed (with CoT): {result_with_cot['policy_id']}")
    # print(f"Saved to: {result_with_cot['file_path']}")
    
    
    # ===================================================================
    # OPTION 2: Process all acceptance policies (uncomment to use)
    # ===================================================================
    
    from utils.data_utils import load_acceptance_policies
    
    # Load all acceptance policies
    acceptance_policies = load_acceptance_policies()
    print(f"Loaded {len(acceptance_policies)} acceptance policies")
    
    # Track success/failure counts
    successful = 0
    failed = 0
    
    # Process each policy (without CoT for bulk processing)
    for policy in acceptance_policies:
        try:
            result = process_policy_to_odrl(
                policy_text=policy["policy_text"],
                policy_id=policy["policy_id"],
                model_name=model_name,
                use_cot_refinement=False
            )
            
            print(f"✓ Processed: {result['policy_id']}")
            successful += 1
            
        except Exception as e:
            print(f"✗ Failed to process {policy['policy_id']}: {str(e)}")
            failed += 1
    
    # Print summary
    print(f"\nProcessing complete. Success: {successful}, Failed: {failed}")