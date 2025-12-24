"""
Enhanced Chain-of-Thought prompts for SIA-LLM.

Key improvements over original prompts:
1. Explicit step-by-step reasoning instructions
2. Self-consistency checks before output
3. Structured JSON output format where beneficial
4. Few-shot examples for complex tasks
5. Context grounding requirements
6. Consistency verification instructions
"""

# =============================================================================
# AGENTS EXTRACTION
# =============================================================================

AGENTS_ENHANCED = """
Let's identify the agents in this scenario step by step.

STEP 1: Read the scenario carefully and identify all characters mentioned.
STEP 2: For each character, determine if they are an active participant (agent) or just mentioned.
STEP 3: Assign valid names (letters, numbers, underscores only - no spaces).

RULES:
- Maximum 3-4 agents for manageable complexity
- Only include characters who take actions in the scenario
- Use underscores instead of spaces in names

SELF-CHECK before output:
- Are all agents actually active in the scenario?
- Do the names follow the naming rules?
- Have I avoided including passive/mentioned-only characters?

Write "The agents are:" and list each agent name between double square brackets.
Example: [[Agent_Name]]
"""

# =============================================================================
# BELIEFS AND DESIRES
# =============================================================================

BELIEFS_DESIRES_ENHANCED = """
Let's generate beliefs and desires for agent [[AGENT NAME]] step by step.

STEP 1: BELIEFS - What does [[AGENT NAME]] know or believe?
Think about:
- Facts [[AGENT NAME]] knows about themselves
- What [[AGENT NAME]] knows about other agents
- [[AGENT NAME]]'s knowledge about the situation
- [[AGENT NAME]]'s assumptions and inferences

STEP 2: DESIRES - What does [[AGENT NAME]] want?
Think about:
- [[AGENT NAME]]'s immediate goals in this scenario
- [[AGENT NAME]]'s motivations and needs
- What outcomes [[AGENT NAME]] is trying to achieve

FORMAT:
- Beliefs: [[BEL(Agent, property) = True/False/Value]]
- Desires: [[DES(Agent, goal) = True/False/Value]]

RULES:
- Each statement must have an explicit value (= True, = False, or = number)
- Use underscores instead of spaces
- Arguments cannot be quoted strings
- Each belief/desire describes exactly ONE thing

SELF-CHECK before output:
- Does every statement have "= Value" at the end?
- Are beliefs grounded in the scenario (not invented)?
- Do desires reflect actual character motivations?
- Is naming consistent (same property name for same concept)?

Example of CORRECT format:
[[BEL([[AGENT NAME]], is_hungry) = True]]
[[BEL([[AGENT NAME]], location) = home]]
[[DES([[AGENT NAME]], eat_food) = True]]

Example of INCORRECT format:
[[BEL([[AGENT NAME]], hungry)]]  <- WRONG: missing = value
[[BEL([[AGENT NAME]], "is hungry") = True]]  <- WRONG: quoted string
"""

# =============================================================================
# INTENTIONS
# =============================================================================

INTENTIONS_ENHANCED = """
Let's derive intentions for agent [[AGENT NAME]] from their beliefs and desires.

STEP 1: Review the beliefs and desires listed above for [[AGENT NAME]].

STEP 2: For each desire, determine what intention would help achieve it.
An intention is a committed goal that [[AGENT NAME]] will actively pursue.

STEP 3: Ensure intentions are achievable given the beliefs.

FORMAT:
[[INTENT(Agent, intention_description) = True]]

RULES:
- Intentions must be achievable given the agent's beliefs
- Each intention should connect to at least one desire
- Use underscores instead of spaces
- Every statement needs "= True" or "= False"

SELF-CHECK before output:
- Is each intention motivated by a desire?
- Can [[AGENT NAME]] realistically pursue this intention given their beliefs?
- Does every statement have "= Value"?
- Are intentions specific enough to plan actions for?

Example:
Given DES([[AGENT NAME]], help_friend) = True
Derive: [[INTENT([[AGENT NAME]], find_and_help_friend) = True]]
"""

# =============================================================================
# ACTION PLAN
# =============================================================================

ACTION_PLAN_ENHANCED = """
Let's create an action plan for intention [[INTENTION]] by agent [[AGENT]] step by step.

STEP 1: Break down the intention into concrete, executable steps.
Think: What specific actions must [[AGENT]] perform to achieve [[INTENTION]]?

STEP 2: Order the actions chronologically.
Think: Which actions must happen before others?

STEP 3: Ensure each action is atomic (one clear thing).

FORMAT:
[[ActionName(AgentPerforming, Target, OptionalArguments)]]

RULES:
- Actions should be in logical order
- Each action should be a single, clear step
- Use consistent naming for actions
- No quoted strings in arguments

SELF-CHECK before output:
- Does the sequence logically lead to achieving the intention?
- Are there any missing steps?
- Is each action executable (not too vague, not too complex)?
- Are actions ordered correctly (dependencies respected)?

Example for INTENT(Alice, make_breakfast):
[[WakeUp(Alice, None)]]
[[GoToKitchen(Alice, kitchen)]]
[[GetIngredients(Alice, fridge, eggs)]]
[[CookFood(Alice, stove, eggs)]]
[[EatBreakfast(Alice, None, eggs)]]
"""

# =============================================================================
# CONDITIONS AND EFFECTS (CRITICAL - Most error-prone)
# =============================================================================

CONDITIONS_EFFECTS_ENHANCED = """
Let's define conditions and effects for action [[ACTION]] by [[AGENT NAME]] carefully.

UNDERSTANDING:
- CONDITIONS: What must be true BEFORE [[AGENT NAME]] can perform [[ACTION]]?
- EFFECTS: What becomes true AFTER [[AGENT NAME]] performs [[ACTION]]?

STEP 1: CONDITIONS
Think about what [[AGENT NAME]] needs:
- What beliefs must [[AGENT NAME]] have?
- What desires motivate this action?
- What state must the world be in?

STEP 2: EFFECTS
Think about what changes:
- What new beliefs does [[AGENT NAME]] gain?
- What beliefs become false?
- What desires are satisfied (become False)?
- What new desires might arise?

FORMAT - CRITICAL:
Conditions:
[[BEL(Agent, property) = Value]]
[[DES(Agent, goal) = Value]]

Effects:
[[BEL(Agent, property) = NewValue]]
[[DES(Agent, goal) = NewValue]]

ABSOLUTE RULES:
1. EVERY statement MUST have "= Value" (True, False, or number)
2. Use ONLY agents that exist in this scenario
3. Use CONSISTENT property names (same name = same concept)
4. No quoted strings in arguments
5. Use underscores instead of spaces

SELF-CHECK before output (DO THIS):
1. Does EVERY line end with "= True", "= False", or "= number"?
2. Are all agent names spelled exactly as defined?
3. Do conditions use beliefs/desires that exist or will be created by prior actions?
4. Do effects create beliefs that later actions might need?
5. Is there logical flow (conditions met -> action -> effects produced)?

EXAMPLE of CORRECT output:
Conditions:
[[BEL([[AGENT NAME]], has_key) = True]]
[[DES([[AGENT NAME]], open_door) = True]]

Effects:
[[BEL([[AGENT NAME]], door_opened) = True]]
[[DES([[AGENT NAME]], open_door) = False]]

EXAMPLE of WRONG output (DO NOT DO THIS):
Conditions:
[[BEL([[AGENT NAME]], has_key)]]  <- WRONG: missing = value
[[BEL(unknown_agent, ready) = True]]  <- WRONG: unknown agent

Write "Conditions:" then list conditions, then "Effects:" then list effects.
"""

# =============================================================================
# DIALOGUE TREE
# =============================================================================

DIALOGUE_TREE_ENHANCED = """
Let's create a dialogue state machine step by step.

STEP 1: Identify the key conversation points between agents.
Think: What are the main things characters need to discuss?

STEP 2: Design states that represent conversation stages.
States should be: Start, topic discussions, responses, End

STEP 3: Create transitions with actual dialogue.

FORMAT:
[[CurrentState, NextState, Meaning, Style, "UtteranceText"]]

COMPONENTS:
- CurrentState: Where conversation is now (letters/numbers only)
- NextState: Where conversation goes next
- Meaning: Topic/intent tag (e.g., Greeting, Question, Answer)
- Style: Personality/tone tag (e.g., Friendly, Formal, Worried)
- UtteranceText: Actual spoken words in quotes

RULES:
- Start from state "Start"
- Include multiple paths (branches) for natural conversation
- Each agent should have speaking opportunities
- Use meaningful state names

SELF-CHECK before output:
- Does dialogue start from "Start" state?
- Are there clear paths to conversation end?
- Do utterances sound natural for the characters?
- Are Meaning and Style filled in (not None)?

Example:
[[Start, Greeting, Greeting, Friendly, "Hello! How are you today?"]]
[[Greeting, AskAboutDay, Question, Curious, "What are you up to?"]]
[[AskAboutDay, SharePlans, Answer, Excited, "I'm going to the park!"]]
"""

# =============================================================================
# SPEAK ACTIONS
# =============================================================================

SPEAK_ACTIONS_ENHANCED = """
Let's identify which dialogue turns agent [[AGENT NAME]] can say.

STEP 1: Review all dialogue turns defined above.

STEP 2: For each turn, determine if [[AGENT NAME]] would be the one speaking.
Consider:
- Is the utterance something [[AGENT NAME]] would say given their personality?
- Does it match [[AGENT NAME]]'s role in the scenario?

STEP 3: List only the dialogue turns that [[AGENT NAME]] performs.

FORMAT:
[[CurrentState, NextState, Meaning, Style, "UtteranceText"]]

SELF-CHECK:
- Are these utterances appropriate for [[AGENT NAME]]'s character?
- Have I avoided assigning the same turn to multiple agents?
"""

# =============================================================================
# SPEAK CONDITIONS EFFECTS
# =============================================================================

SPEAK_CONDITIONS_EFFECTS_ENHANCED = """
Let's define conditions and effects for speak action [[SPEAK ACTION]] by [[AGENT NAME]].

UNDERSTANDING:
- CONDITIONS: What must [[AGENT NAME]] believe/desire to say this?
- EFFECTS: What changes after [[AGENT NAME]] says this?

STEP 1: CONDITIONS
What mental state leads [[AGENT NAME]] to say this?
- What must they believe about the situation?
- What are they trying to achieve?

STEP 2: EFFECTS
What results from saying this?
- Does the listener learn something new?
- Does [[AGENT NAME]]'s state change?
- Are any goals satisfied?

FORMAT:
Conditions:
[[BEL(Agent, property) = Value]]
[[DES(Agent, goal) = Value]]

Effects:
[[BEL(Agent, property) = NewValue]]
[[DES(Agent, goal) = NewValue]]

ABSOLUTE RULES (same as action conditions/effects):
1. EVERY statement MUST have "= Value"
2. Use ONLY agents that exist in this scenario
3. Use CONSISTENT property names

SELF-CHECK:
- Does every line have "= Value"?
- Are effects meaningful (something actually changes)?
- Do conditions make sense for this utterance?

Write "Conditions:" then list, then "Effects:" then list.
"""

# =============================================================================
# INITIAL EMOTION
# =============================================================================

INITIAL_EMOTION_ENHANCED = """
Let's determine [[AGENT NAME]]'s initial emotion using the OCC model.

STEP 1: Consider [[AGENT NAME]]'s situation at scenario start.
- What is happening to them?
- How do they likely feel about it?

STEP 2: Match to an OCC emotion:

POSITIVE EMOTIONS:
- Joy: pleased about a desirable event
- Hope: pleased about prospect of desirable event
- Pride: approving of one's own praiseworthy action
- Admiration: approving of someone else's praiseworthy action
- Happy-for: pleased about event desirable for someone else
- Relief: pleased about disconfirmation of undesirable event
- Satisfaction: pleased about confirmation of desirable event
- Gratitude: approving of someone's action that helped you
- Love: liking an appealing object/person

NEGATIVE EMOTIONS:
- Distress: displeased about undesirable event
- Fear: displeased about prospect of undesirable event
- Shame: disapproving of one's own blameworthy action
- Reproach: disapproving of someone else's blameworthy action
- Pity: displeased about event undesirable for someone else
- Disappointment: displeased about disconfirmation of desirable event
- Fears-confirmed: displeased about confirmation of undesirable event
- Anger: disapproving of someone's action and displeased about result
- Hate: disliking an unappealing object/person

SELF-CHECK:
- Does this emotion fit [[AGENT NAME]]'s initial situation?
- Is it consistent with their beliefs and desires?

Write the emotion between double brackets: [[Emotion]]
"""

# =============================================================================
# INITIAL MOOD
# =============================================================================

INITIAL_MOOD_ENHANCED = """
Let's determine [[AGENT NAME]]'s initial mood on a scale of -10 to 10.

SCALE:
-10 to -6: Very negative (depressed, miserable, hopeless)
-5 to -1: Somewhat negative (sad, frustrated, worried)
0: Neutral
+1 to +5: Somewhat positive (content, hopeful, pleased)
+6 to +10: Very positive (happy, excited, joyful)

STEP 1: Consider [[AGENT NAME]]'s situation at scenario start.
STEP 2: Match to appropriate mood value.

SELF-CHECK:
- Is this mood consistent with the initial emotion?
- Does it fit the scenario context?

Format: [[Mood(SELF)=X]] where X is integer from -10 to 10
"""

# =============================================================================
# ACTION EMOTION APPRAISAL
# =============================================================================

ACTION_EMOTION_ENHANCED = """
Let's determine what [[AGENT NAME]] feels after performing [[ACTION]].

STEP 1: Consider the outcome of [[ACTION]].
- Was it successful?
- How does it affect [[AGENT NAME]]'s goals?

STEP 2: Apply OCC appraisal:
- Is the outcome desirable or undesirable for [[AGENT NAME]]?
- Does [[AGENT NAME]] approve or disapprove of what happened?

STEP 3: Select the appropriate OCC emotion.

Common post-action emotions:
- Pride: after doing something praiseworthy
- Satisfaction: after goal achievement
- Joy: after desirable outcome
- Shame: after doing something blameworthy
- Disappointment: after failed attempt
- Relief: after avoiding negative outcome

SELF-CHECK:
- Does this emotion follow logically from the action outcome?
- Is it consistent with [[AGENT NAME]]'s personality?

Write the emotion between double brackets: [[Emotion]]
"""

# =============================================================================
# EMOTION CONDITION
# =============================================================================

EMOTION_CONDITION_ENHANCED = """
Let's determine what [[AGENT NAME]] must feel BEFORE performing [[ACTION]].

STEP 1: Consider what motivates [[ACTION]].
- Why would [[AGENT NAME]] do this?
- What emotional state leads to this action?

STEP 2: Select appropriate OCC emotion precondition.

Common action-motivating emotions:
- Hope: for actions pursuing positive outcomes
- Fear: for defensive/protective actions
- Anger: for confrontational actions
- Love: for caring/helping actions
- Distress: for escape/avoidance actions

SELF-CHECK:
- Would [[AGENT NAME]] realistically do [[ACTION]] when feeling this emotion?
- Is the emotional motivation consistent with the action's nature?

Write the emotion between double brackets: [[Emotion]]
"""

# =============================================================================
# ACTION MOOD
# =============================================================================

ACTION_MOOD_ENHANCED = """
Let's determine the mood requirement for [[AGENT NAME]] to perform [[ACTION]].

MOOD SCALE: -10 (worst) to +10 (best), 0 = neutral

STEP 1: Consider the nature of [[ACTION]].
- Is it a positive/proactive action? (might require positive mood)
- Is it a defensive/reactive action? (might require negative mood)
- Is it neutral? (no mood requirement)

STEP 2: Define the mood condition.

Options:
- [[Mood(SELF) > X]]: Requires mood above X
- [[Mood(SELF) < X]]: Requires mood below X
- [[Mood(SELF) = X]]: Requires specific mood (rare)

Examples:
- Helping someone: [[Mood(SELF) > 0]] (positive mood needed)
- Complaining: [[Mood(SELF) < 0]] (negative mood needed)
- Routine action: [[Mood(SELF) > -5]] (just not extremely negative)

SELF-CHECK:
- Would [[AGENT NAME]] realistically do [[ACTION]] in this mood?
- Is the threshold reasonable?

Write the mood condition between double brackets.
"""

# =============================================================================
# PROMPT REGISTRY
# =============================================================================

ENHANCED_PROMPTS = {
    "agents": {
        "template": AGENTS_ENHANCED,
        "description": "Enhanced agent extraction with step-by-step reasoning",
    },
    "beliefs_desires": {
        "template": BELIEFS_DESIRES_ENHANCED,
        "description": "Enhanced beliefs/desires with consistency checks",
    },
    "intentions": {
        "template": INTENTIONS_ENHANCED,
        "description": "Enhanced intentions derived from beliefs/desires",
    },
    "action_plan": {
        "template": ACTION_PLAN_ENHANCED,
        "description": "Enhanced action planning with dependency checking",
    },
    "conditions_effects": {
        "template": CONDITIONS_EFFECTS_ENHANCED,
        "description": "Enhanced conditions/effects with strict format validation",
    },
    "dialogue_tree": {
        "template": DIALOGUE_TREE_ENHANCED,
        "description": "Enhanced dialogue tree with branching guidance",
    },
    "speak_actions": {
        "template": SPEAK_ACTIONS_ENHANCED,
        "description": "Enhanced speak action assignment",
    },
    "speak_conditions_effects": {
        "template": SPEAK_CONDITIONS_EFFECTS_ENHANCED,
        "description": "Enhanced conditions/effects for dialogue",
    },
    "initial_emotion": {
        "template": INITIAL_EMOTION_ENHANCED,
        "description": "Enhanced initial emotion selection with OCC guidance",
    },
    "initial_mood": {
        "template": INITIAL_MOOD_ENHANCED,
        "description": "Enhanced initial mood selection",
    },
    "action_emotion": {
        "template": ACTION_EMOTION_ENHANCED,
        "description": "Enhanced post-action emotion appraisal",
    },
    "emotion_condition": {
        "template": EMOTION_CONDITION_ENHANCED,
        "description": "Enhanced pre-action emotion requirement",
    },
    "action_mood": {
        "template": ACTION_MOOD_ENHANCED,
        "description": "Enhanced action mood requirement",
    },
}
