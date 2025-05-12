import pyshacl
from rdflib import Graph
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define ODRL SHACL shapes for validation
basic_ODRL_shape = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix ex: <http://example.org/> .
@prefix drk: <http://w3id.org/drk/ontology/> .

# --- Core Policy Structure Validation ---
ex:PolicyShape a sh:NodeShape ;
    sh:targetClass odrl:Policy ;
    sh:property [
        sh:path rdf:type ;
        sh:minCount 1 ;
        sh:message "Policy must have a type (Agreement, Offer, or Set)" ;
        sh:or (
            [ sh:hasValue odrl:Agreement ]
            [ sh:hasValue odrl:Offer ]
            [ sh:hasValue odrl:Set ]
        )
    ] ;
    sh:property [
        sh:path [sh:alternativePath (odrl:permission odrl:prohibition odrl:obligation)] ;
        sh:minCount 1 ;
        sh:message "Policy must have at least one permission, prohibition, or obligation" ;
    ] .

# --- Rule Shapes ---
# Permission shape - target both explicit types and permission objects
ex:PermissionShape a sh:NodeShape ;
    sh:targetClass odrl:Permission ;
    sh:targetObjectsOf odrl:permission ;
    sh:property [
        sh:path odrl:action ;
        sh:minCount 1 ;
        sh:message "Permission must have an action" ;
    ] ;
    sh:property [
        sh:path odrl:target ;
        sh:minCount 1 ;
        sh:message "Permission must have a target" ;
    ] ;
    # Optional but recommended shape to explicitly type permissions
    sh:property [
        sh:path rdf:type ;
        sh:hasValue odrl:Permission ;
        sh:minCount 0 ;
        sh:severity sh:Info ;
        sh:message "Best practice: Permission objects should be explicitly typed as odrl:Permission" ;
    ] .

# Prohibition shape - target both explicit types and prohibition objects
ex:ProhibitionShape a sh:NodeShape ;
    sh:targetClass odrl:Prohibition ;
    sh:targetObjectsOf odrl:prohibition ;
    sh:property [
        sh:path odrl:action ;
        sh:minCount 1 ;
        sh:message "Prohibition must have an action" ;
    ] ;
    sh:property [
        sh:path odrl:target ;
        sh:minCount 1 ;
        sh:message "Prohibition must have a target" ;
    ] ;
    # Optional but recommended shape to explicitly type prohibitions
    sh:property [
        sh:path rdf:type ;
        sh:hasValue odrl:Prohibition ;
        sh:minCount 0 ;
        sh:severity sh:Info ;
        sh:message "Best practice: Prohibition objects should be explicitly typed as odrl:Prohibition" ;
    ] .

# Obligation/Duty shape - target both explicit types and obligation objects
ex:ObligationShape a sh:NodeShape ;
    sh:targetClass odrl:Duty ;
    sh:targetObjectsOf odrl:obligation ;
    sh:property [
        sh:path odrl:action ;
        sh:minCount 1 ;
        sh:message "Duty must have an action" ;
    ] ;
    # Optional but recommended shape to explicitly type duties
    sh:property [
        sh:path rdf:type ;
        sh:hasValue odrl:Duty ;
        sh:minCount 0 ;
        sh:severity sh:Info ;
        sh:message "Best practice: Obligation objects should be explicitly typed as odrl:Duty" ;
    ] .

# --- Asset Shape ---
ex:AssetShape a sh:NodeShape ;
    sh:targetClass odrl:Asset ;
    sh:targetObjectsOf odrl:target ; # Also validate objects of target property
    sh:property [
        sh:path rdf:type ;
        sh:minCount 1 ;
        sh:message "All assets must be explicitly typed" ;
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:datatype xsd:string ;
        sh:minCount 0 ;
        sh:message "Asset should have a label" ;
    ] .

# --- Party Shape ---
ex:PartyShape a sh:NodeShape ;
    sh:targetClass odrl:Party ;
    sh:targetObjectsOf odrl:assignee, odrl:assigner ; # Also validate objects of party properties
    sh:property [
        sh:path rdf:type ;
        sh:minCount 1 ;
        sh:message "All parties must be explicitly typed" ;
    ] ;
    sh:property [
        sh:path rdfs:label ;
        sh:datatype xsd:string ;
        sh:minCount 0 ;
        sh:message "Party should have a label" ;
    ] .

# --- Constraint Shape ---
ex:ConstraintShape a sh:NodeShape ;
    sh:targetObjectsOf odrl:constraint ;
    sh:property [
        sh:path odrl:leftOperand ;
        sh:minCount 1 ;
        sh:message "Constraint must have a leftOperand" ;
    ] ;
    sh:property [
        sh:path odrl:operator ;
        sh:minCount 1 ;
        sh:message "Constraint must have an operator" ;
    ] ;
    sh:property [
        sh:path [sh:alternativePath (odrl:rightOperand odrl:rightOperandReference)] ;
        sh:minCount 1 ;
        sh:message "Constraint must have either a rightOperand or rightOperandReference" ;
    ] ;
    # Optional but recommended shape to explicitly type constraints
    sh:property [
        sh:path rdf:type ;
        sh:hasValue odrl:Constraint ;
        sh:minCount 0 ;
        sh:severity sh:Info ;
        sh:message "Best practice: Constraint objects should be explicitly typed as odrl:Constraint" ;
    ] .

# --- Metadata Validation ---
ex:MetadataShape a sh:NodeShape ;
    sh:targetClass odrl:Policy ;
    sh:property [
        sh:path odrl:uid ;
        sh:minCount 1 ;
        sh:message "Policy must have a unique identifier (uid)" ;
    ] .
"""

def validate_with_pyshacl(data_ttl: str, shapes_ttl: str) -> dict:
    """
    Validate RDF data against SHACL shapes using PyShacl
    
    Args:
        data_ttl (str): The policy/data in Turtle format
        shapes_ttl (str): The SHACL shapes in Turtle format
        
    Returns:
        dict: A dictionary with validation results
    """
    try:
        # Parse data graph
        data_graph = Graph()
        data_graph.parse(data=data_ttl, format="turtle")
        
        # Parse shapes graph
        shapes_graph = Graph()
        shapes_graph.parse(data=shapes_ttl, format="turtle")
        
        # Perform validation
        logger.info("Running SHACL validation with PyShacl")
        conforms, results_graph, results_text = pyshacl.validate(
            data_graph=data_graph,
            shacl_graph=shapes_graph,
            inference='rdfs',
            abort_on_first=False
        )
        
        logger.info(f"Validation results: {'Conforms' if conforms else 'Does not conform'}")
        
        # Process violations
        violations = []
        if not conforms:
            # Create a set to track unique violation messages
            unique_violation_keys = set()
            
            # Convert results to more usable format
            for result in results_graph.subjects(predicate=None, object=None):
                message = None
                focus_node = None
                
                # Find message and focus node for this result
                for pred, obj in results_graph.predicate_objects(subject=result):
                    if "resultMessage" in str(pred):
                        message = str(obj)
                    elif "focusNode" in str(pred):
                        focus_node = str(obj)
                
                if message and focus_node:
                    # Create a unique key based on message and focus node
                    violation_key = f"{message}|{focus_node}"
                    
                    # Only add if this exact violation hasn't been seen before
                    if violation_key not in unique_violation_keys:
                        unique_violation_keys.add(violation_key)
                        violations.append({
                            "message": message,
                            "node": focus_node
                        })
        
        return {
            "conforms": conforms,
            "violations": violations,
            "results_text": results_text
        }
        
    except Exception as e:
        logger.error(f"Error in SHACL validation: {str(e)}")
        return {
            "conforms": False,
            "violations": [{"message": f"Error validating data: {str(e)}"}],
            "results_text": f"Validation failed: {str(e)}"
        }

def validate_odrl_policy(policy_ttl, shapes_ttl=None):
    """
    Validate an ODRL policy against SHACL shapes
    
    Args:
        policy_ttl (str): The ODRL policy in Turtle format to validate
        shapes_ttl (str): Optional SHACL shapes in Turtle format (uses built-in shapes if None)
    
    Returns:
        dict: Validation results with conformance status and violations
    """
    # Use default shapes if none provided
    if shapes_ttl is None:
        shapes_ttl = basic_ODRL_shape
    
    # Validate the policy
    results = validate_with_pyshacl(policy_ttl, shapes_ttl)
    
    return results

# --- Azure OpenAI Configuration ---
AZURE_API_KEY = "xxxxx"
AZURE_ENDPOINT = "xxx.com"

AZURE_API_VERSION = "2024-10-01-preview"
DEPLOYMENT_NAME = "gpt-4o-mini-2024-07-18"

# --- LLM Setup ---
llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME,
)


from typing import Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Proper state definition using TypedDict
class PolicyFixerState(TypedDict):
    """State for the policy fixer agent"""
    policy_text: str  # Natural language policy description
    policy_ttl: str  # ODRL TTL policy to fix
    validation_results: Optional[Dict]  # SHACL validation results
    fixed_policy_ttl: Optional[str]  # Fixed policy

# Node to validate the policy
def validate_policy(state: PolicyFixerState) -> PolicyFixerState:
    """Validate the policy using the SHACL validator"""
    logger.info("Validating ODRL policy using validator")
    
    # Use the validation function defined above
    validation_results = validate_odrl_policy(state["policy_ttl"])
    
    logger.info(f"Validation complete. Policy conforms: {validation_results['conforms']}")
    if not validation_results['conforms']:
        logger.info(f"Found {len(validation_results['violations'])} validation issues")
    
    # Return updated state
    return {
        **state,
        "validation_results": validation_results
    }

# Node to fix the policy using LLM
def fix_policy(state: PolicyFixerState) -> PolicyFixerState:
    """Use LLM to fix the policy based on validation results"""
    logger.info("Using LLM to fix policy based on validation results")
    
    # Format validation violations separately
    violations_text = []
    for v in state["validation_results"]["violations"]:
        violations_text.append(f"- {v['message']} (Node: {v['node']})")
    violations_str = "\n".join(violations_text)
    
    # Create prompt for the LLM
    prompt = f"""You are an expert in ODRL policies and SHACL validation. 
Your task is to fix an ODRL policy based on SHACL validation results.

POLICY DESCRIPTION IN NATURAL LANGUAGE:
{state["policy_text"]}

CURRENT ODRL POLICY (TTL FORMAT):
```turtle
{state["policy_ttl"]}
```

VALIDATION RESULTS:
Conforms: {state["validation_results"]["conforms"]}
Violations:
{violations_str}

Please fix all the validation issues and return the corrected ODRL policy in TTL format.
Only return the fixed TTL policy, no explanations or additional text.
"""
    
    # Get response from LLM
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    fixed_policy = response.content
    
    # Extract the policy if it's in a code block
    if "```" in fixed_policy:
        parts = fixed_policy.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Every other part is inside the code blocks
                if part.startswith("turtle"):
                    fixed_policy = part[6:].strip()  # Remove "turtle" marker
                else:
                    fixed_policy = part.strip()
                break
    
    # Save the fixed policy to a file
    with open("fixed_policy.ttl", "w") as f:
        f.write(fixed_policy)
        
    logger.info("Fixed policy saved to fixed_policy.ttl")
    
    # Return updated state
    return {
        **state,
        "fixed_policy_ttl": fixed_policy
    }

# Build the agent
def build_policy_fixer_agent():
    """Build and compile the policy fixer agent"""
    # Create state graph with our state class
    graph = StateGraph(PolicyFixerState)
    
    # Add nodes
    graph.add_node("validate", validate_policy)
    graph.add_node("fix", fix_policy)
    
    # Add edges - simple linear flow
    graph.add_edge("__start__", "validate")
    graph.add_edge("validate", "fix")
    graph.add_edge("fix", END)
    
    # Compile the graph
    return graph.compile()

# Main function to run the agent
def run_agent(policy_text: str, policy_ttl: str):
    """Run the agent on a policy"""
    # Create the agent
    agent = build_policy_fixer_agent()
    
    # Create initial state as a dictionary
    initial_state: PolicyFixerState = {
        "policy_text": policy_text,
        "policy_ttl": policy_ttl,
        "validation_results": None,
        "fixed_policy_ttl": None
    }
    
    # Run the agent
    logger.info("Starting policy validation and fixing process")
    result = agent.invoke(initial_state)
    
    # Print the fixed policy
    logger.info("Process completed")
    print("\nFIXED POLICY:")
    print(result["fixed_policy_ttl"])
    
    # Show original validation issues
    print("\nORIGINAL VALIDATION ISSUES:")
    for v in result["validation_results"]["violations"]:
        print(f"- {v['message']} (Node: {v['node']})")
    
    return result

# Example usage
if __name__ == "__main__":
    # Example policy text
    policy_text = "Access to the dataset titled 'MuseumArtifact' is permitted only to approved dataspace participants(UC4) operating through authorized connectors. Only participants using connector endpoints registered as '?connector1URI' are allowed to use the data."
    
    # Example policy with issues (missing uid)
    example_policy = """@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix drk:   <http://w3id.org/drk/ontology/> .

drk:DatenRaumkulturImageAccessPolicy a odrl:Agreement ;
    dct:title "Daten Raumkultur Image Access Policy" ;
    dct:description "Policy allowing UC4 partners to view a maximum of 50 images per day" ;
    dct:issued "2023-10-01"^^xsd:date ; # Assume a recent issuance date
    odrl:permission [
        odrl:action odrl:view ;
        odrl:target drk:DatenRaumkulturImageDataset ;
        odrl:assigner drk:DatenRaumkulturConnector ;
        odrl:assignee drk:UC4Partners ;
        odrl:constraint [
            odrl:leftOperand odrl:count ;
            odrl:operator odrl:lteq ;
            odrl:rightOperand "50"^^xsd:integer
        ]
    ] .

drk:DatenRaumkulturImageDataset a odrl:Asset, dct:Dataset ;
    dct:title "Daten Raumkultur Image Dataset" ;
    dct:description "Dataset containing images related to Raumkultur" .

drk:DatenRaumkulturConnector a odrl:Party, foaf:Organization ;
    rdfs:label "Daten Raumkultur Connector Organization" .

drk:UC4Partners a odrl:Party, drk:Group ;
    rdfs:label "UC4 Partner Organizations" .
"""
    
    # Run the agent
    run_agent(policy_text, example_policy)
