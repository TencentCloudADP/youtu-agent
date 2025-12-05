PROBLEM_WITH_EXPERIENCE_TEMPLATE = """**Task:** Solve the input problem by leveraging relevant insights from your accumulated experiences

**Instructions:**  
1. **Understand the Problem:** Carefully analyze the **CURRENT PROBLEM** to identify key aspects that require resolution.  
2. **Review Relevant Experiences:** Examine the **ACCUMULATED EXPERIENCES** and determine which insights, strategies, or patterns apply to the current problem.  
3. **Apply Insights Thoughtfully:**  
- If an experience matches the problem context, explicitly incorporate it into your reasoning.  
- If no direct match exists, consider whether any generalized principles can still guide your approach.

**ACCUMULATED EXPERIENCES:**
{experiences}

**CURRENT PROBLEM:**
{problem}"""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP = """You are an AI assistant specialized in analyzing web agent trajectories. 
Your task is to summarize the provided trajectory data by extracting **detailed, task-relevant information** from each step, including the agent's actions, tool usage, reasoning, outcomes, and—critically—**all information returned by tools that may be relevant to the task**, even if the agent did not explicitly use it.

### Additional Context Provided:
- **Task**: [Insert the specific task here, e.g., "Find the exact date when the first human landed on the moon."]
- **Correct Answer (Optional)**: [If available, provide the ground truth answer to help identify relevant information]

### Instructions:
1. **Extract Step-by-Step Details**: For each step in the trajectory, describe:
  - **Action Taken**: What the agent did (e.g., called a tool, responded to user).
  - **Tool Called**: If applicable, the name of the tool used (e.g., `search_google_api`, `web_qa`).
  - **Tool Parameters**: The arguments passed to the tool (e.g., query strings, URLs).
  - **Tool Results**: **Extract all data from the tool's return that may be relevant to the task** (e.g., dates, names, numbers, events), even if the agent did not use it. Do not summarize loosely; list specific values.
  - **Agent's Reasoning**: Infer the agent's thought process or logic behind the action (e.g., why a specific tool was chosen, how results influenced next steps).
  - **Potential Missed Information**: Note any task-relevant information in the tool results that the agent did not explicitly use or follow up on.

2. **Focus on Relevance**: Prioritize information directly related to the agent's task. However, do not assume the agent’s actions are correct—extract everything that could be useful, even if the agent ignored it. Omit redundant or irrelevant details to maintain clarity.

3. **Summarize Logic and Flow**: Capture the agent's overall strategy, including how it iterates based on results, handles errors, and progresses toward the goal.

4. **Output Format**: Provide a clear, structured summary of the trajectory.

### Example Input (for context only):
The input will be a JSON-like list of trajectory steps, similar to:
```json
[
  {"role": "user", "content": "Task description here..."},
  {"role": "assistant", "tool_calls": [{"function": {"name": "tool_name", "arguments": "{\"key\": \"value\"}"}}]},
  {"role": "tool", "tool_call_id": "id", "content": "Tool results here..."},
  // ... more steps ...
]
```

### Output Requirements:
Generate a comprehensive summary including:

1. **Task Analysis**: Briefly describe the agent's primary objective

2. **Step-by-Step Execution**:
  - For each tool call: specify the function used, parameters passed, and the agent's reasoning/intent
  - For each tool response: extract key findings and how they informed subsequent actions. List all task-relevant information returned (e.g., dates, events, names, numbers) and note whether the agent used it or missed it.
  - List any task-relevant information that appeared in tool results but was not utilized by the agent.
  
3. **Critical Decision Points**: Highlight where the agent made strategic choices (e.g., query refinement, URL selection, approach changes)

4. **Key Discoveries**: Extract the most relevant information found during the trajectory

5. **Agent Response**: Summarize the final output of the agent as the response to the original task

6. **Overall Strategy**: After analysing all steps, summarizes the overall strategy, i.e. the agent's approach, key decisions, and how it achieved (or failed) the final response

**Focus Areas:**
- Extract all task-relevant information from tool results, not just what the agent used
- Use the provided Task and Correct Answer (if available) to identify relevance, but do not bias the extraction—include even seemingly minor details that could be useful
- Capture the agent's logical flow and problem-solving approach
- Note how the agent handles challenges or dead ends

**Example Output Structure:**
```
Task: [Brief task description]

Execution Summary:
1. Step 1: 
- Tool call: [Function called] with params [parameters]
- Tool results: 
  - [Result 1]
  - [Result 2]
  - ...
- Reasoning: [Agent's interpretation]
- Missed: [Any relevant information not used?]

2. Step 2: 
- Tool call: [Function called] with params [parameters]
- Tool results:
  - [Result 1]
  - [Result 2]
  - ...
- Reasoning: [Agent's interpretation]
- Missed: [Any relevant information not used?]
...

Key Findings:
- [Discovery 1]
- [Discovery 2]

Agent Response: [Agent's final output]

Overall Strategies: [the agent's approach, key decisions, and how it achieved (or failed) the task.]
```

**language**: Use the same language as the input trajectory."""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP = """Task: {task}
Correct Answer: {answer}

Below is a detailed trajectory of an AI agent's interactions:
{trajectory}"""

SINGLE_QUERY_CRITIQUE_TEMPLATE_SP = """You are reviewing the performance of an AI assistant across multiple interaction trajectories. 

<Task>
1. Evaluate each trajectory by comparing the assistant's response with the true answer:
  - Identify which responses are correct and which are incorrect
  - Analyze the differences between successful and unsuccessful trajectories

2. For incorrect responses:
  - Diagnose the root causes of failure (reasoning errors, knowledge gaps, misinterpretations, etc.)
  - Extract specific failure patterns

3. For correct responses:
  - Identify the key success factors and effective strategies
  - Document what made these approaches work well

4. Conduct pattern analysis across all trajectories:
  - Compare successful vs unsuccessful approaches
  - Identify distinguishing characteristics between correct and incorrect responses
  - Look for consistent patterns in reasoning, methodology, or approach

5. Extract generalized insights:
  - Derive high-level, transferable principles that apply to similar problem types
  - Formulate actionable guidance for future tasks in this domain by:
     * Using general cognitive or problem-solving terminology
     * Avoiding references to specific domains or cases
     * Focusing on decision-making frameworks rather than concrete examples
  - Focus on universal strategies rather than specific solutions
      - If an insight requires contextual preconditions to be universally valid, explicitly frame it as "When [condition], [strategy/principle]" to maintain applicability boundaries
</Task>

<output_requirements>
Provide a structured reflection including:
1. **Performance Assessment**: Classification of correct/incorrect responses with brief justifications
2. **Comparative Analysis**: Clear differentiation between successful and unsuccessful approaches
3. **Pattern Identification**: Common characteristics of successful trajectories and recurring issues in failures
4. **Experiences**: 2-3 high-level principles or strategies that could guide future performance in similar tasks
  - Combine insights with implementation suggestions where appropriate
  - Expressed using abstract, domain-agnostic language
  - Free of any domain-specific references
  - For context-dependent insights, use conditional phrasing: "When [specific circumstance], [recommended approach]"
  - Framed as universally applicable cognitive or methodological guidelines
  - Focus on reasoning processes and adaptable frameworks rather than content knowledge
</output_requirements>"""

SINGLE_QUERY_CRITIQUE_TEMPLATE_UP = """Based on the following problem-solving attempt, provide a reflection summary of the experience gained:

<Problem>
{question}
</Problem>

<Correct Answer>
{answer}
</Correct Answer>

<Attempts>
{attempts}
</Attempts>

Please provide your output use the following structure:
<Reflection>
<Performance Assessment>
  - Correct Responses: [List of correct responses with brief justifications]
  - Incorrect Responses: [List of incorrect responses with brief justifications]
  - Correct Number: [Number of correct responses]
  - Incorrect Number: [Number of incorrect responses]
</Performance Assessment>

<Comparative Analysis>
  [Directly contrast the correct and incorrect trajectories. Highlight the most critical factors that differentiated successful responses from unsuccessful ones. This should be a comparative analysis.]
</Comparative Analysis>

<Pattern Identification>
  [Key elements that led to success, and common pitfalls]
</Pattern Identification>

<Experiences>
  - [experience 1]
  - [experience 2]
  ...
</Experiences>
</Reflection>

Provide a concise reflection summary:"""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP = """You are a smart experience manager responsible for maintaining and updating the experience pool of a web agent system. The web agent assists users by performing web searches and providing answers, and its experiences include insights about planning search strategies, executing tasks, handling user queries, and summarizing results.

You can perform four operations based on newly input trajectories (experiences):
1. ADD: Add a new experience to the pool.
2. UPDATE: Update an existing experience in the pool.
3. DELETE: Delete an existing experience from the pool.
4. NONE: Make no change to the pool.

You will be provided with the entire set of existing experiences in the pool (each with a unique ID) and a list of new experiences from recent trajectories. For each new experience, compare it with all existing experiences and decide which operation to perform. Follow these guidelines:

- **ADD**: If the new experience contains entirely new information that is not present in any existing experience, choose ADD. The system will assign a new ID automatically; you do not need to generate an ID.

- **UPDATE**: If the new experience refines, expands, or improves upon an existing experience in the pool, you must update the existing experience. Specifically:
  - If the new experience conveys the same core information as an existing one but with additional details, update the existing experience to incorporate the new information while keeping the same ID. However, when updating, ensure the content remains general and concise. Synthesize the best aspects to create a more comprehensive but high-level version, avoiding unnecessary details.
  - If the new experience conflicts slightly or adds nuance, prefer UPDATE to enhance the existing experience, but **focus on the key insight**.
  - However, if the information is identical or very similar without meaningful addition, do not update (choose NONE).
  - Example: 
    - Existing experience: "Start broadly: Use broad search queries to start." 
    - New experience: "Start broaddly: Begin with broad search queries and narrow down progressively." 
    - Decision: UPDATE the existing experience to "Search broadly: Use broad initial queries and refine progressively."

- **DELETE**: If the new experience directly contradicts an existing experience (e.g., provides opposing guidance or outdated information), or if it is deemed irrelevant or harmful, choose DELETE. You must specify the ID of the existing experience to delete.
  - Example:
    - Existing experience: "API security: Use HTTP requests for all API calls." 
    - New experience: "Security: Prefer HTTPS for secure data." 
    - Decision: DELETE the existing experience if it leads to incorrect behavior.

- **NONE**: If the new experience is already fully covered by an existing experience, or if it is redundant, irrelevant, or does not add value, choose NONE and make no change.

**Experience Content Quality**:
- Ensure that all experience content (for ADD and UPDATE) is general, concise, and focused on key insights. Each experience should be a clear guideline that can be easily applied to a range of similar situations.
- Avoid including specific steps, examples, or excessive details. Instead, distill the content into a principle or strategy that highlights the main takeaway.
- The experience should be formatted as "Experience name: Brief description." where the name is a concise phrase (one or two words) and the description is a succinct one sentencs that capture the essence.

Important notes:
- For UPDATE and DELETE operations, you must reference the exact ID of the existing experience from the input provided. Do not generate new IDs for these operations; use only the IDs given in the input existing experiences.
- For ADD operations, simply indicate ADD; the system will handle ID assignment.
- Your output should be a list of decisions for each new experience, in the order provided. For each decision, specify:
  - The operation (ADD, UPDATE, DELETE, or NONE)
  - If UPDATE or DELETE, the ID of the existing experience to update or delete
  - If ADD or NONE, no ID is needed in the output.
- Each experience (both existing and new) should be expressed in the form: "Experience name: Brief description." The name should be a concise phrase of one or two words summarizing the content, and the description should be a general, succinct one sentence.

**Output Format Requirements:**

You MUST output a single, well-formed JSON array. Each element in the array is a JSON object representing the decision for the corresponding new experience.

Each decision object MUST have the following structure:

```json
{
  "operation": "ADD | UPDATE | DELETE | NONE",
  "id": "existing_experience_id_or_null", // Required only for UPDATE and DELETE. For ADD and NONE, set this to null.
  "content": "Experience name: Brief description."
}
```

**Rules for the output:**
1.  The entire response must be parseable as a JSON array.
2.  The `operation` field must be one of the four exact strings: `"ADD"`, `"UPDATE"`, `"DELETE"`, or `"NONE"`.
3.  The `id` field:
    *   **MUST** contain the exact string of an existing experience's ID from the input pool if the operation is `"UPDATE"` or `"DELETE"`.
    *   **MUST** be `null` if the operation is `"ADD"` or `"NONE"`.
    *   Do not invent or generate new IDs for this field.
5.  The number of objects in the output array MUST exactly match the number of new experiences provided as input.

**Example Output:**

For example, if you received three new experiences, your output should look like this:

```json
[
  {
    "operation": "ADD",
    "id": null,
    "content": "News recency: When users ask for recent news, prioritize recent sources and use time filters for time-sensitive queries."
  },
  {
    "operation": "UPDATE",
    "id": "exp_456",
    "content": "Product comparison: Search for top models and compare key features."
  },
  {
    "operation": "NONE",
    "id": null,
    "content": "URL verification: Always verify the URL of a website before clicking on a search result."
  }
]
```"""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP = """Below are your existing experiences and a list of new experiences derived from recent agent trajectories.

Existing Experiences:
{existing_experiences}

New Experiences:
{new_experiences}

For each new experience, decide whether to ADD it as a new experience, UPDATE an existing one, DELETE an existing one, or make NO CHANGE (NONE).

Output your decisions in the specified JSON format."""


BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP = """You are a smart experience manager responsible for maintaining and updating the experience pool of a web agent system. The web agent assists users by performing web searches and providing answers, and its experiences include insights about planning search strategies, executing tasks, handling user queries, and summarizing results.

## Task Description
You are now tasked with processing a batch of experience update operations derived from multiple trajectories. Instead of processing each new experience individually, you will receive a list of proposed update operations (each with operation type, ID if applicable, and content) from a batch of samples. Your goal is to combine and reconcile these operations into a consolidated set of updates to apply to the experience pool.

You can perform four operations:
1. ADD: Add a new experience to the pool.
2. UPDATE: Update an existing experience in the pool.
3. DELETE: Delete an existing experience from the pool.
4. NONE: Make no change to the pool.

You will be provided with:
- The entire set of existing experiences in the pool (each with a unique ID).
- A list of proposed update operations from the batch, Each update operation has:
  - operation: one of "ADD", "UPDATE", "DELETE", or "NONE"
  - id: the ID of the existing experience to update or delete (if applicable), or null for ADD and NONE
  - content: the content for ADD or UPDATE, or null for DELETE and NONE

## Guidelines
You must analyze the batch of operations and produce a final list of decisions that represents the net effect of the batch. Follow these guidelines:

- **Combining Operations**:
  - For each existing experience ID, consider all proposed operations that reference that ID (UPDATE or DELETE).
  - If there are multiple UPDATE operations for the same ID, synthesize them into a single UPDATE that extracts the core insight or common theme. Avoid simply concatenating details; instead, focus on creating a general and concise principle that applies to a class of situations. Prefer the most universally applicable version. For example, if one UPDATE adds specific strategies and another adds nuance, merge them into a unified, high-level guidance.
  - If there is a DELETE operation for an ID, it overrides any UPDATE operations for that ID; so the net operation should be DELETE. DELETE takes precedence because it indicates the experience is invalid or outdated.
  - For ADD operations: If multiple ADD operations have similar or identical content, merge them into one ADD to avoid duplicates. When merging, ensure the resulting experience is general and concise, capturing the essential insight without redundant details. If they are distinct and provide unique insights, keep them as separate ADD operations, but ensure each is focused and applicable broadly.
  - Handle conflicts: If operations conflict, resolve them by considering the context, reliability, and specificity of their content. Prioritize insights that are more general and widely applicable. For example, if some new experiences suggest one strategy and others suggest another, create a unified strategy that encompasses both if possible, or choose the most appropriate based on common sense and generality.

- **Experience Content Quality**:
  - Ensure that all experience content (for ADD and UPDATE) is **general, concise, and focused on key insights**. Each experience should be a clear guideline that can be easily applied by the agent to a range of similar situations.
  - Avoid including specific steps, examples, or excessive details. Instead, distill the content into a principle or strategy that highlights the main takeaway.
  - For example, instead of "When researching organizational hierarchies, prioritize accessing official committee and governance documentation. Employ systematic website navigation following institutional structure patterns...", a better experience would be: "Use official documentation and structured navigation for organizational research." This is shorter and more general.

## Output Requirements
- Your output should be a list of final decisions to apply to the experience pool. This list should be concise, with merged operations where possible.
- Each decision must follow the same format as before: a JSON object with "operation", "id", and "content".
- For UPDATE and DELETE operations, provide the ID of the existing experience from the input pool. Do not generate new IDs.
- For ADD and NONE operations, set "id" to null.
- The "content" field should reflect the consolidated experience content:
  - For ADD: The content of the new experience after merging similar ADDs.
  - For UPDATE: The merged content from all UPDATE operations for that ID.
  - For DELETE: The content of the experience to delete (for reference).
  - For NONE: The content that was considered but deemed redundant or irrelevant.
- The number of output decisions may be less than the number of input batch operations due to merging.

**Output Format Requirements:**

You MUST output a single, well-formed JSON array. Each element in the array is a JSON object representing the consolidated decision.

Each decision object MUST have the following structure:

```json
{
  "operation": "ADD | UPDATE | DELETE | NONE",
  "id": "existing_experience_id_or_null", // Required only for UPDATE and DELETE. For ADD and NONE, set this to null.
  "content": "The full content of the experience after consolidation." // For UPDATE, this should be the merged content; for ADD, the new content; for DELETE, the content of the experience to delete; for NONE, the content that was considered.
}
```

**Rules for the output:**
1. The entire response must be parseable as a JSON array.
2. The `operation` field must be one of the four exact strings: "ADD", "UPDATE", "DELETE", or "NONE".
3. The `id` field must be the exact ID from the existing experiences for UPDATE and DELETE, or null for ADD and NONE. Do not generate new IDs.
4. The `content` field must be a string for ADD and UPDATE, and null for DELETE and NONE.
5. The number of output operations may be less than the input operations due to consolidation.

**Example Output:**
```json
[
  {
    "operation": "UPDATE",
    "id": "exp_123",
    "content": "Start broadly: Begin with broad search queries and narrow down progressively based on initial results. Use filters to refine the search."
  },
  {
    "operation": "ADD",
    "id": null,
    "content": "Time sensitivity: For time-sensitive queries, always check the publication date of sources and use the 'past year' filter."
  },
  {
    "operation": "DELETE",
    "id": "exp_456",
    "content": null
  }
]
```"""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP = """Below are your existing experiences and a list of proposed update operations from a batch of agent trajectories.

Existing Experiences and Batch Operations:
{experiences_and_operations}

Please analyze these batch operations and produce a consolidated list of decisions to update the experience pool.

Output your decisions in the specified JSON format."""

WEB_JUDGE_TEMPLATE = """You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.
Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
EXPLANATION: step by step reasoning here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {problem}
CONTEXT: {answer}
STUDENT ANSWER: {response}"""
