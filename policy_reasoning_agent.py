from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, TypedDict, Dict, Any, Union, Optional
import logging
import os
import json
import datetime
import re
from policy_reasoning_utils import extract_confidence_score


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Azure OpenAI Configuration ---
AZURE_API_KEY = "xxxxx"
AZURE_ENDPOINT = "xxx.com"

# API versions - uncomment one
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

# --- LangGraph State Definition with proper typing ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    initial_analysis: str
    final_analysis: str
    needs_rethinking: bool
    iterations: int
    enable_rethinking: bool
    confidence_score: Optional[float]
    reasoning_chain: List[Dict[str, Any]]  # Store each step of reasoning

# --- LLM Setup ---
llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME,
)

# --- System Prompts ---
analysis_prompt = """
You are a highly intelligent policy reasoning agent that processes natural language policy statements and performs concrete, high-accuracy reasoning. Your tasks include:

Core Tasks:

1. **Policy Type Identification:**
   - Identify the **Policy Type** based on the following rules:
     - **Agreement**: A policy where both an assigner and assignee are explicitly mentioned
       * An Agreement MUST have one `assigner` property value (of type Party) 
       * An Agreement MUST have one `assignee` property value (of type Party)
     - **Offer**: A proposed arrangement where an assigner offers terms to a potential assignee
       * An Offer MUST have one `assigner` property value (of type Party)
       * An Offer MAY have an `assignee` property value (of type Party)
     - **Set**: A collection of defined rules or prohibitions with no explicit parties mentioned
       * Default policy type when no parties are identified
   
   - NOTE: Vague party definitions are ACCEPTABLE. The policy should NOT be rejected solely because parties are not precisely defined.
       
   - Within the policy type, identify the **Rule Type** as one of:
     - **Permission**: Allows an action ("can", "may", "is allowed to")
     - **Prohibition**: Forbids an action ("cannot", "must not", "is prohibited from")
     - **Obligation**: Requires an action ("must", "shall", "is required to")
     - **Duty**: States what should happen if obligations are not met
     
2. **Asset Identification:**
   - Identify the **Target Assets** affected by the policy.
   - If no assets are explicitly mentioned in the policy:
     * Issue a STRONG WARNING that "This policy does not clearly identify target assets"
     * Suggest potential assets based on context (e.g., "data," "records," "resources")
     * DO NOT REJECT the policy solely for missing or vague assets
     * Instead, mark it as NEEDS REVISION and recommend adding specific asset identification

3. **Action Extraction & Actor Identification:**
   - Extract **Actions** (e.g., read, share, delete).
   - CRITICAL: All actions MUST be specific, clear, and measurable. 
   - IMMEDIATELY REJECT policies with vague action verbs such as:
     * "handle", "manage", "deal with", "take care of"
     * "process", "review", "consider", "evaluate" (without specific criteria)
     * "support", "assist", "help", "facilitate" (without specific actions)
     * "improve", "enhance", "optimize" (without measurable criteria)
   - Actions must have clear beginning and end states, and must be objectively observable.

4. **Constraint Identification:**
   - Identify any **constraints** related to the policy, including:
     - **Temporal constraints** (e.g., deadlines, durations, time windows)
     - **Location constraints** (e.g., geographic region, country-specific restrictions)
     - **Role-based constraints** (e.g., only users with role "editor" may edit)
     - **Quantitative or count-based constraints** (e.g., "no more than 3 times")
   
   - For **temporal constraints**:
     - Current date/time is {current_time}
     - IMMEDIATELY REJECT ANY POLICY containing:
       * Expired temporal constraints (dates already passed)
       * ANY non-measurable time references, including but not limited to:
         - "business hours", "working hours", "off-peak hours"
         - "regularly", "periodically", "occasionally", "frequently"
         - "soon", "shortly", "quickly", "promptly"
         - "when necessary", "as needed", "as appropriate"
         - "during the academic year", "during the season"
         - "reasonable time", "adequate time"
       * CRITICAL: Overlapping time windows with contradictory rules, such as:
         - If access is granted from 9AM-5PM but also prohibited from 2PM-6PM
         - If a resource must be available 24/7 but also undergo scheduled maintenance
         - If an action must be completed within 48 hours but cannot be started until 36 hours after triggering
     - All temporal constraints MUST have exact times, dates, or precisely defined durations (e.g., "within 24 hours", "30 days from receipt")
  
   - For **location constraints**:
     - IMMEDIATELY REJECT ANY POLICY containing:
       * Vague location references like:
         - "appropriate regions", "suitable locations", "where applicable"
         - "relevant territories", "nearby areas"
       * CRITICAL: Location hierarchy contradictions such as:
         - If access is prohibited in Country X but required in City Y, which is within Country X
         - If a resource must be accessible in European Union but forbidden in Germany
         - If data must be stored in North America but cannot be stored in the United States
     - All locations must be specifically named (e.g., "Germany", "Berlin") or precisely defined
     - When locations have containment relationships, rules must be consistent across the hierarchy
   
   - For **role-based constraints**:
     - IMMEDIATELY REJECT ANY POLICY containing:
       * Vague role references like:
         - "authorized personnel", "qualified individuals"
         - "appropriate stakeholders", "relevant parties"
       * CRITICAL: Role hierarchy contradictions such as:
         - If Managers cannot access a resource but Department Heads (who are also Managers) must access it
         - If all Employees are forbidden from an action but a specific Employee role is required to perform it
         - If a role has both an obligation and a prohibition for the same action/asset
     - All roles must be specifically defined with clear responsibilities
     - When roles have inheritance relationships, rules must be consistent across the hierarchy

   - For **quantitative constraints**:
     - IMMEDIATELY REJECT ANY POLICY containing vague quantity references like:
       * "some", "several", "many", "few"
       * "reasonable amount", "appropriate quantity"
       * "minimal", "sufficient", "adequate"
       * "as much as needed", "as required"
     - All quantitative constraints MUST specify exact numbers or percentages
     - CRITICAL: Check for contradictory quantity limits (e.g., maximum 10 uses but also minimum 20 uses)

5. **Semantic Validation:**
   - **Action Conflict Detection:** IMMEDIATELY REJECT any policy that contains directly contradictory rules:
     * Permission and Prohibition for the same action on the same asset by the same actor
     * Obligation to perform an action that is also prohibited
     * Rules requiring mutually exclusive states or conditions
   
   - **Role Consistency:** Ensure actors are assigned roles consistent with their actions.
   
   - **Validity of Actors/Assets:** Verify that the assets referenced are valid.
   
   - **Ensure Completeness:** Ensure that all elements like Assigner, Assignee, and action targets are complete.
   
   - **DECISION PATH COMPLETENESS CHECK:**
      * Identify all decision points in the policy (approve/deny, accept/reject, etc.)
      * For each decision point, verify that all possible outcomes have defined handling procedures
      * Check if any decision outcomes are mentioned but lack subsequent processing steps
      * Example: "Requests may be approved or denied, but only approved requests have defined handling"
      * IMMEDIATELY REJECT policies with incomplete decision paths as they create implementation gaps

   - **CRITICAL CIRCULAR DEPENDENCY CHECK:**
     * Map out all prerequisite relationships in the policy (A requires B, B requires C, etc.)
     * Trace each dependency chain to check if any path leads back to itself
     * Example of circular dependency: "Action A requires condition B → condition B requires action C → action C requires condition A"
     * Real example: "Researchers need committee approval → approval requires verification → verification requires preliminary access → preliminary access requires committee approval"
     * IMMEDIATELY REJECT any policy with circular dependencies as they create implementation impossibilities

   - **DATA RETENTION CONTRADICTION CHECK:**
     * Identify all data deletion requirements (what data must be deleted and when)
     * Identify all data retention requirements (what data must be kept and for how long)
     * Check for overlapping data types that have both deletion and retention requirements
     * EXAMPLES OF CONTRADICTIONS:
       - "All user data must be deleted after 90 days BUT user purchase history must be retained for 5 years"
       - "All system logs must be purged monthly BUT security audit trails containing user activities must be maintained for 3 years"
     * IMMEDIATELY REJECT policies that require both deletion and retention of the same data without a resolution mechanism (such as pseudonymization, data segregation, or explicit exemption clauses)

   - **CRITICAL SEMANTIC CONTRADICTION CHECKS:**
     * For each combination of actor, action, and asset in the policy:
       - List all rules that apply to this combination
       - Check if any rules contradict each other (permission vs. prohibition, etc.)
       - Check if any rules create impossible situations (e.g., must do X and must not do X)
     * For each asset in the policy:
       - Check if contradictory states are required (e.g., data must be both encrypted and unencrypted)
     * IMMEDIATELY REJECT any policy with semantic contradictions as they make implementation impossible
     
6. **Real-world Applicability Assessment:**
   - Determine if the policy can be reasonably implemented and enforced in a real-world setting.
   - CRITICAL CHECK: Identify if the policy uses universal quantifiers ("everyone", "everything", "all", "any") without appropriate scope limitations.
   - Identify if the policy is overly broad, vague, or impractical for actual implementation.
   - Consider whether the policy would create unintended consequences or implementation challenges.
   - Evaluate whether the policy would violate basic security principles, privacy requirements, or regulatory constraints.
   - IMMEDIATELY REJECT policies that grant universal access without appropriate restrictions as they are fundamentally incompatible with real-world security requirements.

7. **Decision Criteria:**
   Determine the appropriate decision for the policy based on these criteria:
   
   - APPROVED: The policy is clear, specific, can be implemented with no or minor issues, and is applicable in real-world settings.
   
   - NEEDS REVISION: The policy has moderate issues that can be addressed with specific changes, EXCEPT for the critical issues listed under REJECTED criteria.
   
   - REJECTED: The policy MUST be rejected without exception if ANY of these conditions are true:
     * It contains circular dependencies where prerequisites form a loop that cannot be satisfied
     * It contains expired temporal constraints (dates that have already passed)
     * It contains ANY non-measurable temporal constraints (see detailed list above)
     * It contains overlapping time windows with contradictory requirements
     * It contains location-based rules that create geographical hierarchy contradictions
     * It contains role-based rules that create role hierarchy contradictions
     * It contains ANY vague action verbs (see detailed list above)
     * It contains ANY non-measurable quantitative constraints (see detailed list above)
     * It contains ANY vague location or role-based constraints (see detailed lists above)
     * It uses universal quantifiers or extremely broad terms like "anything," "everything", "nobody", "everybody", "everyone", "all" without adequate specificity
     * It has semantic contradictions (e.g., permission and prohibition for the same action/asset/actor)
     * It would be impossible to implement in a real-world setting

   **CRITICAL DECISION FLOW:**
   ALWAYS follow this exact decision flow in sequence:
   1. First, check for ANY rejection triggers:
      * Non-measurable constraints - Any term that lacks objective, quantifiable criteria, including:
         - Temporal terms without specific timeframes (e.g., "urgent", "reasonable time", "promptly", "soon", "periodically", "regularly", "when appropriate")
         - Qualitative terms without defined thresholds (e.g., "adequate", "sufficient", "appropriate", "satisfactory", "good quality", "effectively")
         - Subjective judgment terms (e.g., "important", "significant", "meaningful", "relevant", "suitable")
         - Conditional terms without clear criteria (e.g., "if necessary", "when needed", "as required", "if appropriate")
      
      * Vague action verbs - Any verb that lacks clear beginning/end states or objective verification, including:
         - Knowledge processing verbs without criteria (e.g., "review", "evaluate", "assess", "consider", "analyze", "examine")
         - Management verbs without specific actions (e.g., "manage", "handle", "oversee", "administer", "coordinate", "deal with")
         - Support verbs without defined activities (e.g., "support", "assist", "help", "facilitate", "enable", "aid")
         - Improvement verbs without measurable outcomes (e.g., "improve", "enhance", "optimize", "upgrade", "refine")
         - Communication verbs without specific channels/methods (e.g., "inform", "notify", "communicate", "advise" without specifying how)
   
   2. If ANY rejection trigger is present, the decision MUST be "REJECTED"
   3. If NO rejection triggers are present but moderate issues exist, the decision is "NEEDS REVISION"
   4. Only if NO rejection triggers AND NO moderate issues exist, the decision is "APPROVED"

   This decision flow is MANDATORY and overrides any other considerations.
   
   **CRITICAL RULES:**
   - Vague or undefined parties and assets should NOT cause rejection - mark these as NEEDS REVISION instead.
   - ANY policy containing logical contradictions, non-measurable constraints, vague actions, or universal quantifiers MUST be REJECTED, not marked for revision.
   - Policies that would be impossible to implement in real-world systems due to contradictions, security, privacy, or practical concerns MUST be REJECTED.
Analyze the following policy statement:
{user_input}
"""

rethinking_prompt = """
You are a critical policy reasoning validator. Your task is to carefully review the initial analysis of a policy statement using a step-by-step chain of thought approach. You will methodically evaluate each aspect of the analysis to identify errors, omissions, or potential improvements.

Original policy statement: {user_input}

Initial analysis: {initial_analysis}

## Chain of Thought Reasoning Process

Follow these steps sequentially, providing your explicit reasoning at each step:

STEP 1 - Policy Type Validation:
- What policy type was identified in the initial analysis? (Agreement, Offer, Request, or Set)
- Carefully re-examine the original policy statement.
- Is this identification correct? Why or why not?
- If incorrect, what should the correct policy type be and why?
- NOTE: Vague party definitions are ACCEPTABLE and should NOT cause policy rejection.

STEP 2 - Rule Type Validation:
- What rule type was identified? (Permission, Prohibition, Obligation, or Duty)
- Does the policy statement use language that clearly matches this rule type?
- Is there a more appropriate rule type classification? Explain your reasoning.

STEP 3 - Action Extraction Verification:
- List all actions identified in the initial analysis.
- Are there any actions missed in the initial analysis? If so, what are they?
- Are any identified actions not actually present in the policy? Explain.
- CRITICAL CHECK: Are any actions vague or non-measurable? Actions like "handle", "manage", "process", "review", "support", "assist", "improve", or "optimize" without specific criteria are unacceptable and MUST trigger REJECTION.
- All actions must be specific, clear, and objectively observable.

STEP 4 - Actor Identification Verification:
- Who were identified as the Assigner and Assignee?
- Is this identification correct and complete? Why or why not?
- If incorrect, who should be correctly identified as the Assigner and Assignee?
- NOTE: Vague actor definitions should lead to NEEDS REVISION, not REJECTION.

STEP 5 - Target Asset Verification:
- What target assets were identified in the initial analysis?
- Are all relevant assets from the policy statement accounted for?
- Are there any assets incorrectly identified or missing?
- NOTE: Vague or missing assets should lead to NEEDS REVISION, not REJECTION.

STEP 6 - Constraint Analysis:
- What constraints were identified in the initial analysis?
- For each constraint, is it properly validated for measurability?
- CRITICAL CHECK for TEMPORAL CONSTRAINTS:
  * Identify any temporal references in the policy
  * Check if they use non-measurable terms like "business hours", "periodically", "soon", etc.
  * Check for overlapping time windows with contradictory rules
  * Example: "Access allowed 9AM-5PM but prohibited 2PM-6PM creates a contradiction"
- CRITICAL CHECK for LOCATION CONSTRAINTS:
  * Identify any location references in the policy
  * Check if they use vague terms like "appropriate regions", "suitable locations", etc.
  * Check for location hierarchy contradictions
  * Example: "Access prohibited in Germany but required in Berlin (which is in Germany)"
- CRITICAL CHECK for ROLE-BASED CONSTRAINTS:
  * Identify any role references in the policy
  * Check if they use vague terms like "authorized personnel", "qualified individuals", etc.
  * Check for role hierarchy contradictions
  * Example: "Managers cannot access but Department Heads (who are Managers) must access"
- CRITICAL CHECK for QUANTITATIVE CONSTRAINTS:
  * Identify any quantity references in the policy
  * Check if they use vague terms like "some", "several", "many", "few", etc.
  * Check for contradictory quantity limits
  * Example: "Maximum 10 uses but minimum 20 uses"
- ANY non-measurable constraint or contradiction MUST trigger immediate REJECTION.

STEP 7 - Semantic Validation Review:
- Were all potential conflicts between actions properly identified?
- DECISION PATH COMPLETENESS CHECK:
  * Are there decision points in the policy (approve/deny, accept/reject, etc.)?
  * Do all possible outcomes from these decision points have defined handling procedures?
  * Example: "Requests may be approved or denied, but only approved requests have defined handling"
- CIRCULAR DEPENDENCY CHECK:
  * Are there prerequisite relationships in the policy?
  * Do any dependency chains lead back to themselves creating a loop?
  * Example: "A requires B → B requires C → C requires A"
- DATA RETENTION CONTRADICTION CHECK:
  * Are there both deletion and retention requirements for the same types of data?
  * Example: "Personal data deleted after 30 days BUT transaction records with user details kept for 7 years"
- Is role consistency maintained throughout the analysis?
- Are all mentioned assets valid within the context?
- Is the analysis complete with all necessary elements?

STEP 8 - Hierarchy Violation Detection:
- Are there any location or jurisdiction hierarchies that might conflict?
- For locations mentioned, are there any containment relationships (e.g., city within country)?
- For roles mentioned, are there any inheritance relationships (e.g., managers are also employees)?
- Have any potential hierarchy conflicts been properly identified and explained?

STEP 9 - Real-world Applicability Assessment:
- Is this policy actually implementable in a real-world setting?
- Would it be technically, logistically, or practically feasible to enforce this policy?
- Are there unintended consequences or implementation challenges that were not identified?
- Would this policy cause unreasonable burden or be impossible to follow?
- Does the policy use universal quantifiers ("everyone", "everything", "all", "any") without appropriate scope limitations?

STEP 10 - Comprehensive Assessment:
- Based on steps 1-9, what aspects of the initial analysis are correct and comprehensive?
- What specific issues need correction or additional analysis?
- How would you improve the analysis based on the issues identified?

STEP 11 - DECISION CRITERIA:
   - Determine the appropriate decision for the policy based on these criteria:
   
   - APPROVED: The policy is clear, specific, can be implemented with no or minor issues, and is applicable in real-world settings.
   
   - NEEDS REVISION: The policy has moderate issues that can be addressed with specific changes, EXCEPT for the critical issues listed under REJECTED criteria.
   
   - REJECTED: The policy MUST be rejected without exception if ANY of these conditions are true:
     * It contains circular dependencies where prerequisites form a loop that cannot be satisfied
     * It contains incomplete decision paths where critical outcomes lack defined handling procedures
     * It contains data retention contradictions requiring both deletion and retention of the same data
     * It contains expired temporal constraints (dates that have already passed)
     * It contains ANY non-measurable temporal constraints
     * It contains overlapping time windows with contradictory requirements
     * It contains location-based rules that create geographical hierarchy contradictions
     * It contains role-based rules that create role hierarchy contradictions
     * It contains ANY vague action verbs without specific criteria
     * It contains ANY non-measurable quantitative constraints
     * It contains ANY vague location or role-based constraints
     * It uses universal quantifiers or extremely broad terms without adequate specificity
     * It has semantic contradictions (e.g., permission and prohibition for the same action/asset/actor)
     * It would be impossible to implement in a real-world setting
     * It contains obligations without activation conditions or temporal triggers
    - Example: "Employees must submit expense reports" (When? After what event?)
    - Example: "The system must archive communications" (At what point? Under what circumstances?)
    - All obligations REQUIRE explicit activation conditions (e.g., "after travel", "when requested", "within 14 days")
    - Untriggered obligations make enforcement impossible as
   
   CRITICAL RULE: Any policy containing logical contradictions, non-measurable constraints, vague actions, or universal quantifiers MUST be REJECTED, not marked for revision.
   CRITICAL RULE: Vague parties and assets alone should NOT cause rejection - these issues should only trigger NEEDS REVISION status.
   
## Final Determination

Based on your step-by-step analysis above, provide your final determination:
1. If the analysis is completely correct and comprehensive, begin with: "The analysis is satisfactory."
2. If you identified any issues or improvements, provide a corrected analysis that addresses all the issues you found.
3. Make a clear decision on whether the policy should be APPROVED, NEEDS REVISION, or REJECTED.
4. If your decision is to REJECT the policy, provide a completely rewritten alternative that addresses all identified issues.
"""

user_friendly_output_prompt = """
Based on the thorough analysis of the policy statement:

"{user_input}"

Please create a user-friendly output with the following clearly defined sections:

## Original Policy Statement
[Include the exact policy statement]

## Policy Analysis Summary
- Policy Type: [Identify as Agreement, Offer, Request, or Set]
- Rule Type: [Identify as Permission, Prohibition, Obligation, or Duty]
- Actions: [List the actions identified in the policy]
- Actors: [Identify who the Assignee and Assigner are]
- Target Assets: [List what resources or data are affected]

## Issues Detected
[List all issues clearly, categorized by type]
1. Non-Measurable Constraint Issues (CRITICAL - REJECTION TRIGGERS):
   - [Identify ANY non-measurable constraints that require immediate rejection]
   - [For each one, explain exactly why it's non-measurable and how it violates requirements]

2. Vague Action Issues (CRITICAL - REJECTION TRIGGERS):
   - [Identify ANY vague actions that require immediate rejection]
   - [For each one, explain exactly why it's vague and how it violates requirements]

3. Temporal Contradiction Issues (CRITICAL - REJECTION TRIGGERS):
   - CRITICAL - TEMPORAL CONFLICTS: Check if any time periods contradict each other by creating a timeline:
     * List all time windows (Start time - End time) that apply to the same actors and assets
     * For each pair of overlapping windows, check if the rules contradict (e.g., one allows while another prohibits)
     * Example: "This policy allows access from 9AM-5PM but prohibits access from 2PM-6PM, creating a direct contradiction during 2PM-5PM"
     * Any such contradiction MUST trigger REJECTION

4. Location Hierarchy Issues (CRITICAL - REJECTION TRIGGERS):
   - CRITICAL - LOCATION HIERARCHY CONFLICTS: Check if location requirements contradict by analyzing containment:
     * List all locations mentioned in the policy with their hierarchical relationships
     * Check if rules applied to a location conflict with rules applied to containing locations
     * Example: "This policy prohibits access in Germany but requires access in Berlin, which is contained within Germany"
     * Any such contradiction MUST trigger REJECTION

5. Ambiguity Issues (May Require Revision):
   - [Describe each ambiguity with recommendations to fix]

6. Completeness Issues (May Require Revision):
   - [List missing elements that should be added]
   - [Note: Vague parties and assets fall here and should NOT trigger rejection]

7. Conflict Issues:
   - [Describe any other logical conflicts or contradictions not covered in the specific categories above]
   - CRITICAL - CIRCULAR DEPENDENCY CHECK: Check if the policy contains a process where each step requires another step that eventually loops back to itself:
     * Example: "Approval requires verification → verification requires submission → submission requires approval"
     * Such circular dependencies make implementation IMPOSSIBLE and MUST trigger REJECTION
8. Hierarchy Violations:
   - [Identify any other hierarchy violations not covered in the specific categories above]
   - For location hierarchies, explicitly state relationships like "X is part of Y"
   - For role hierarchies, explicitly state relationships like "X role includes Y role"

9. Real-world Implementation Issues:
   - Determine if the policy can be reasonably implemented and enforced in a real-world setting
   - CRITICAL CHECK: Identify if the policy uses universal quantifiers ("everyone", "everything", "all", "any") without appropriate scope limitations
   - Identify if the policy is overly broad, vague, or impractical for actual implementation
   - Consider whether the policy would create unintended consequences or implementation challenges
   - Evaluate whether the policy would violate basic security principles, privacy requirements, or regulatory constraints
   - IMMEDIATELY REJECT policies that grant universal access without appropriate restrictions as they are fundamentally incompatible with real-world security requirements

## Decision
[Make a clear decision on whether an ODRL policy can be created or should be rejected]
- Status: [APPROVED/NEEDS REVISION/REJECTED]
- Rationale: [Brief explanation of the decision]

IMPORTANT DECISION CRITERIA:
   - APPROVED: The policy is clear, specific, can be implemented with no or minor issues, and is applicable in real-world settings.
   
   - NEEDS REVISION: The policy has moderate issues that can be addressed with specific changes, EXCEPT for the critical rejection issues. Vague parties and assets should trigger NEEDS REVISION, not REJECTION.
   
   - REJECTED: The policy MUST be rejected without exception if ANY of these conditions are true:
     * It contains overlapping time windows with contradictory requirements for the same actors/assets
     * It contains location-based rules that conflict due to geographical hierarchy (e.g., one rule for a city, a conflicting rule for the country containing it)
     * It contains expired temporal constraints (dates that have already passed)
     * It contains ANY non-measurable constraints such as:
        - Temporal: "business hours", "working hours", "regularly", "periodically", "soon", etc.
        - Quantitative: "some", "several", "many", "few", "reasonable amount", etc.
        - Location-based: "appropriate regions", "suitable locations", etc.
        - Role-based: "authorized personnel", "qualified individuals", etc.
     * It contains vague actions such as "handle", "manage", "process", "review" without specific criteria
     * It uses vague terms like "anything," "everything", "nobody", "everybody" without specificity
     * It has critical semantic contradictions
     * It would be impossible to implement in a real-world setting
      If ANY process or approval flow creates a loop where:
       - A requires B → B requires C → C requires A (or any longer chain that loops back)
    * It contains obligations without activation conditions or temporal triggers
    - Example: "Employees must submit expense reports" (When? After what event?)
    - Example: "The system must archive communications" (At what point? Under what circumstances?)
    - All obligations REQUIRE explicit activation conditions (e.g., "after travel", "when requested", "within 14 days")
    - Untriggered obligations make enforcement impossible as
     **MANDATORY VALIDATION CHECK:**
  Before finalizing your analysis, double-check:
  1. Did you identify ANY non-measurable constraint or vague action verb? If YES, the policy MUST be REJECTED.
  2. Did you identify ANY logical contradiction or implementation impossibility? If YES, the policy MUST be REJECTED.
  3. Is your final decision consistent with the issues you identified? If not, correct your decision now.
    
   CRITICAL RULE: Non-measurable constraints, vague actions, or logical contradictions MUST trigger REJECTION, not NEEDS REVISION.
   CRITICAL RULE: Vague parties and assets alone should NOT cause rejection.
   
## Alternative Policy
[If the policy is REJECTED, provide a completely rewritten alternative policy that addresses all identified issues and would be feasible to implement]

## Improvement Suggestions
[If the policy is marked NEEDS REVISION, provide specific suggestions to improve it]

Your analysis and reasoning: {analysis}
"""

# --- Abstracted LLM Call Function ---
def call_llm(prompt: str) -> str:
    """Centralized function for all LLM calls to ensure consistency and easier updating"""
    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

def analyze_policy(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content  # Extract last message content
    
    logger.info("Starting initial policy analysis")
    
     # Get current time as a string directly in the function
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    # Create the analysis prompt with user input and current time
    prompt = analysis_prompt.format(user_input=user_input, current_time=current_time)
    
    # Invoke LLM
    response = call_llm(prompt)
    
    # Extract confidence score (internally used for rethinking decision)
    confidence = extract_confidence_score(response)
    
    # Store the initial analysis
    state["initial_analysis"] = response
    state["confidence_score"] = confidence
    
    # Initialize reasoning chain
    state["reasoning_chain"] = [{
        "iteration": 0,
        "stage": "initial_analysis",
        "full_response": response,
        "confidence": confidence
    }]
    
    # Only set needs_rethinking to True if enable_rethinking is True
    # and confidence is below threshold (0.9)
    if state["enable_rethinking"] and confidence < 0.9:
        state["needs_rethinking"] = True
    else:
        state["needs_rethinking"] = False
    
    state["iterations"] = 0
    
    logger.info("Initial analysis complete")
    
    return state

# --- Single Rethinking Node (No Recursion) ---
def rethink_analysis(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content
    initial_analysis = state["initial_analysis"]
    
    logger.info("Starting chain of thought rethinking")
    
    # Create the rethinking prompt with chain of thought structure
    prompt = rethinking_prompt.format(user_input=user_input, initial_analysis=initial_analysis)
    
    # Invoke LLM for structured rethinking
    response = call_llm(prompt)
    
    # Extract confidence score (internally used)
    confidence = extract_confidence_score(response)
    state["confidence_score"] = confidence
    
    # Store the chain of thought reasoning
    cot_reasoning = {
        "iteration": 1,
        "full_response": response,
        "confidence": confidence,
    }
    
    # Add this reasoning step to our chain
    state["reasoning_chain"].append(cot_reasoning)
    
    # Always proceed to final output after one rethinking step
    state["final_analysis"] = response
    state["iterations"] = 1
    
    logger.info("Chain of thought rethinking complete")
    
    return state

# --- Final Output Node ---
def generate_final_output(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content
    
    final_analysis = state["final_analysis"] if state["final_analysis"] else state["initial_analysis"]
    reasoning_chain = state.get("reasoning_chain", [])
    
    logger.info("Generating final output with chain of thought reasoning")
    
    if state["iterations"] > 0:
        format_prompt = f"""
        Based on the thorough analysis of the policy statement with chain of thought reasoning:
        
        "{user_input}"
        
        Please create a user-friendly output with the following clearly defined sections:
        
        ## Original Policy Statement
        [Include the exact policy statement]
        
        ## Policy Analysis Summary
        - Policy Type: [Identify as Agreement, Offer, Request, or Set]
        - Rule Type: [Identify as Permission, Prohibition, Obligation, or Duty]
        - Actions: [List the actions identified in the policy]
        - Actors: [Identify who the Assignee and Assigner are]
        - Target Assets: [List what resources or data are affected]
        
        ## Chain of Thought Reasoning
        The initial analysis was refined through chain of thought reasoning.
        
        Key improvements from the chain of thought process:
        - Identify 2-3 significant improvements made during the reasoning process
        - Highlight how the final analysis differs from the initial assessment
        - Explain which aspects required the most critical re-examination
        
        ## Issues Detected
        [List all issues clearly, categorized by type]
        1. Ambiguity Issues:
           - [Describe each ambiguity with recommendations to fix]
        2. Measurability Issues:
           - [Describe each measurable constraint problem with specific improvements]
        3. Completeness Issues:
           - [List missing elements that should be added]
        4. Conflict Issues:
           - [Describe any logical conflicts or contradictions]
        5. Hierarchy Violations:
           - [Specifically identify any location/region contradictions, with clear explanation of the hierarchy relationships]
           - For location hierarchies, explicitly state relationships like "X is part of Y"
           - For role hierarchies, explicitly state relationships like "X role includes Y role"
        6. Real-world Implementation Issues:
           - [Identify any practical barriers to implementing this policy in real-world settings]
           - [Explain why the policy might be unreasonable or impossible to enforce]
        
        ## Decision
        [Make a clear decision on whether an ODRL policy can be created or should be rejected]
        - Status: [APPROVED/NEEDS REVISION/REJECTED]
        - Rationale: [Brief explanation of the decision]
        
        IMPORTANT DECISION CRITERIA:
        - A policy should be APPROVED only if it is clear, specific, has no critical issues, and can be reasonably implemented.
        - A policy should be marked NEEDS REVISION if it has moderate issues that can be addressed with specific changes.
        - A policy should be REJECTED if:
          * It uses extremely vague terms like "anything," "everything," "nobody," "everybody" without specificity
          * It lacks clear actors, actions, or target assets
          * It contains unmeasurable constraints that cannot be reasonably defined
          * It has critical semantic contradictions
          * It would be impossible or unreasonable to implement in a real-world setting
        
        ## Alternative Policy
        [If the policy is REJECTED, provide a completely rewritten alternative policy that addresses all identified issues and would be feasible to implement]
        
        ## Improvement Suggestions
        [If the policy is marked NEEDS REVISION, provide specific suggestions to improve it]
        
        Your analysis and reasoning: {final_analysis}
        """
    else:
        # Use the original format without chain of thought section
        format_prompt = user_friendly_output_prompt.format(
            user_input=user_input, 
            analysis=final_analysis
        )
    
    # Invoke LLM for final structured output
    response = call_llm(format_prompt)
    
    # Add the final output to messages
    state["messages"].append(AIMessage(content=response))
    
    logger.info("Final output with chain of thought generated successfully")
    
    return state

# --- Route Based on Rethinking Flag ---
def route_after_analysis(state: AgentState) -> str:
    if state["enable_rethinking"] and state["confidence_score"] < 0.9:
        return "rethink"
    else:
        return "finalize"

# --- Define LangGraph ---
def build_policy_reasoning_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("analyze", analyze_policy)
    builder.add_node("rethink", rethink_analysis)
    builder.add_node("finalize", generate_final_output)
    
    builder.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "rethink": "rethink",
            "finalize": "finalize"
        }
    )
    
    builder.add_edge("rethink", "finalize")
    builder.set_entry_point("analyze")
    builder.set_finish_point("finalize")
    return builder.compile()

# --- Compile Graph ---
graph = build_policy_reasoning_graph()

def analyze_policy_reasoning(user_input: str, enable_rethinking: bool = False, show_cot: bool = True):
    print(f"\n--- Running test case: {user_input} ---")
    print(f"--- Rethinking enabled: {enable_rethinking} | Show Chain of Thought: {show_cot} ---\n")
    
    messages = [HumanMessage(content=user_input)]
    result = graph.invoke({
        "messages": messages,
        "initial_analysis": "",
        "final_analysis": "",
        "needs_rethinking": False,
        "iterations": 0,
        "enable_rethinking": enable_rethinking,
        "confidence_score": None,
        "reasoning_chain": []
    })
    if show_cot and "reasoning_chain" in result and len(result["reasoning_chain"]) > 1:
        print("\n--- Chain of Thought Reasoning Process ---\n")
        
        initial = result["reasoning_chain"][0]
        print("Initial Analysis")
        print("=" * 50)
        print(initial['full_response'][:500] + "..." if len(initial['full_response']) > 500 else initial['full_response'])
        print("=" * 50)
        
        # Show rethinking step
        cot = result["reasoning_chain"][1]
        print("\nRethinking Analysis")
        print("=" * 50)
        print(cot['full_response'][:500] + "..." if len(cot['full_response']) > 500 else cot['full_response'])
        print("=" * 50)
    
    print("\n--- Agent Final Output ---\n")
    # Print only the final AI message
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
            break
    
    print(f"\n--- Total Iterations: {result['iterations']} ---\n")
    
    return result  
  
def evaluate_policies_with_reasoning(policy_dataset,
    expected_decision,  # "APPROVED" or "REJECTED"
    model_name="gpt-4o-mini",
    output_dir="agent_reasoning_results",
    enable_rethinking=True
):
    """
    Simple function to evaluate policies and save results for manual review.
    
    Args:
        policy_dataset: List of policy dictionaries to evaluate
        expected_decision: What decision we expect ("APPROVED" or "REJECTED")
        model_name: Name of the model being used
        output_dir: Base directory for results
        enable_rethinking: Whether to enable rethinking feature
    """
    # Create output directory with model name and expected decision subdirectory
    model_dir = os.path.join(output_dir, model_name)
    save_dir = os.path.join(model_dir, expected_decision)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n===== EVALUATING {len(policy_dataset)} POLICIES WITH {model_name} =====")
    print(f"Expected decision: {expected_decision}")
    print(f"Rethinking enabled: {enable_rethinking}")
    
    # Initialize counters
    counts = {"APPROVED": 0, "REJECTED": 0, "NEEDS REVISION": 0, "UNKNOWN": 0}
    
    # Process each policy
    for i, policy in enumerate(policy_dataset):
        policy_id = policy["policy_id"]
        policy_text = policy["policy_text"]
        
        print(f"\n[{i+1}/{len(policy_dataset)}] Processing: {policy_id}")
        
        # Call the reasoning agent
        result = analyze_policy_reasoning(
            policy_text, 
            enable_rethinking=enable_rethinking, 
            show_cot=False
        )
        
        # Find LLM's response
        response = None
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                response = msg.content
                break
        
        if not response:
            print(f"WARNING: No response for policy {policy_id}")
            counts["UNKNOWN"] += 1
            continue
        
        # Extract decision if possible
        decision_match = re.search(r"Status:\s*(APPROVED|REJECTED|NEEDS REVISION)", response)
        llm_decision = decision_match.group(1) if decision_match else "UNKNOWN"
        
        # Update counter
        if llm_decision in counts:
            counts[llm_decision] += 1
        else:
            counts["UNKNOWN"] += 1
        
        # Create output with manual review placeholders
        output = f"""# Policy Analysis: {policy_id}

## Original Policy
{policy_text}

## Expected Decision
{expected_decision}

## LLM's Decision
{llm_decision}

## Manual Review
[ ] Correct (LLM matches expected)
[ ] Incorrect (LLM differs from expected)
[ ] Override to: _______________

## Bellow is LLMs reasoning for this decision
---------------------------------------
{response}
"""
        
        # Save to file
        filename = f"{policy_id}.txt"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)
        
        print(f"LLM decision: {llm_decision} | Expected: {expected_decision}")
    
    # Create summary JSON exactly as requested
    summary = {
        "model_name": model_name,
        "total_policies": len(policy_dataset),
        "approved/reviewed_count": counts["APPROVED"],
        "rejected_count": counts["REJECTED"],
        "expected_decision": expected_decision,
        "rethinking_enabled": enable_rethinking
    }
    
    # Save summary
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary

if __name__ == "__main__":
  from utils.data_utils import load_rejection_policies, load_acceptance_policies
  # For rejection dataset
  data = [
 {
        "policy_id": "drk_ambiguous_condition_policy",
        "policy_text": "If researchers submit access requests for restricted manuscripts, then curators will review the request. If the request is approved, users can download high-resolution images. If the request is urgent, it will be expedited.",
        "expected_outcome": "REJECTED",
        "rejection_category": "conditional_ambiguity",
        "specific_contradiction": "Policy fails to define what constitutes an 'urgent' request, leaving a critical decision point unmeasurable",
        "recommendation": "Define specific, measurable criteria for what qualifies as an 'urgent' request (e.g., needed within 48 hours, required for pending publication)",
        "rejection_reason_detailed": "The policy contains an unmeasurable condition ('urgent') that affects processing priority. Without specific criteria for urgency, implementation would be inconsistent and subjective. The policy should define measurable criteria for classifying requests as urgent to ensure consistent handling."
    },
  ]
  
  # gpt-35-turbo gpt-35-turbo-0125
  # gpt-4o-mini gpt-4o-mini-0125
  # gpt-4o gpt-4o-2024-11-20 
  # o1-mini	o1-mini-2024-09-12 unsupported_value
  # o3-mini	o3-mini-2025-01-31
 
  evaluate_policies_with_reasoning(
        # policy_dataset=load_rejection_policies(),
        policy_dataset= data,
        expected_decision="REJECTED",
        model_name="o3-mini",
        enable_rethinking=False
    )
  
  # # For Accepted dataset
  # evaluate_policies_with_reasoning(
  #       policy_dataset=load_acceptance_policies(),
  #       expected_decision="APPROVED",
  #       model_name="gpt-4o-mini",
  #       enable_rethinking = False
  #   )

 