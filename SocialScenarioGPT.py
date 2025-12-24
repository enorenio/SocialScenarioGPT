import re
from tqdm import tqdm
import time
import json
import os

from Constants.api_contants import API_KEY
from Constants.task_constants import *
from OpenAIHandler import OpenAIHandler

# Feature flag imports for ablation study (TASK-014)
from config.feature_flags import FeatureFlags, get_profile, PROFILES
from models import ModelFactory, get_model
from prompts import get_prompt_manager
from core.verification import verify_conditions_effects, VerificationResult
from core.error_feedback import format_errors_for_llm, format_regeneration_prompt

NUM_TRIES = 3

# Global feature flags instance - controls which features are active
# Default is baseline (all features off)
_active_flags: FeatureFlags = FeatureFlags()

def set_feature_flags(flags: FeatureFlags) -> None:
    """Set the active feature flags for scenario generation."""
    global _active_flags
    _active_flags = flags

def get_feature_flags() -> FeatureFlags:
    """Get the currently active feature flags."""
    return _active_flags

# This should be a variable asked to be inputed by the user but for now is a constant here
scenario_description = "Social scenario of a bartender with two customers."

def describe_task(model, task_description):
    model.add_user_turn(task_description)


def scenario_task(model, scenario_description, task_description):
    task_description = re.sub(re.escape("[[DESCRIPTION HERE]]"), scenario_description, task_description)
    model.add_user_turn(task_description)
    #response = model.get_model_response()

    # Remove turns to save input space
    #model.remove_turns(-1)
    #model.add_model_turn(response)

    #return  response.choices[0].message.content.strip()


def get_agents_task(model, task_description):
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    model.add_model_turn(response)

    # Parse agents
    agents = response.choices[0].message.content.strip()
    agents = re.findall("\[\[.*?\]\]", agents)

    return agents


def get_beliefs_desires(model, agent, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse beliefs and desires
    bel_des_base = response.choices[0].message.content.strip()
    bel_des_base = re.findall("\[\[.*?\]\]", bel_des_base)
    bel_des_base = [re.sub(",[ ]+?\)]", "\)", elem) for elem in bel_des_base if elem != agent]

    # print(response.choices[0].message.content.strip())

    return bel_des_base

def get_intentions(model, agent, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse beliefs and desires
    intentions = response.choices[0].message.content.strip()
    intentions = re.findall("\[\[.*?\]\]", intentions)
    intentions = [re.sub(",[ ]+?\)]", "\)", elem) for elem in intentions if elem != agent]

    # print(response.choices[0].message.content.strip())

    return intentions



def get_initial_mood(model, agent, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    mood = response.choices[0].message.content.strip()
    mood = re.findall("\[\[.*?\]\]", mood)
    mood = [re.sub("Mood\(.+?\)", "Mood(SELF)", elem) for elem in mood if elem != agent]

    return mood

def get_agent_actions(model, agent, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse actions
    actions = response.choices[0].message.content.strip()
    actions = re.findall("\[\[.*?\]\]", actions)
    actions = [re.sub(",[ ]+?\)]", "\)", elem) for elem in actions if elem != agent and "ActionName" not in elem]

    # print(response.choices[0].message.content.strip())

    return actions


def _parse_conditions_effects(response_text: str, agent: str, action: str):
    """Parse conditions and effects from LLM response."""
    try:
        conditions = re.findall("[\s\S]+?[Ee]ffects", response_text)[0]
        conditions = re.findall("\[\[.*?\]\]", conditions)
        conditions = [re.sub(",[ ]+?\)]", "\)", elem) for elem in conditions if elem != action and elem != agent]
    except:
        conditions = []
    try:
        effects = re.findall("[Ee]ffects[\s\S]+", response_text)[0]
        effects = re.findall("\[\[.*?\]\]", effects)
        effects = [re.sub(",[ ]+?\)]", "\)", elem) for elem in effects if elem != action and elem != agent]
    except:
        effects = []
    return conditions, effects


def get_action_conditions_effects(model, agent, action, task_description,
                                   domain_knowledge=None, max_retries=3):
    """
    Get conditions and effects for an action.

    If verification_loop feature is enabled and domain_knowledge is provided,
    will verify the output and request regeneration if errors are found.
    """
    flags = get_feature_flags()
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    task_description = re.sub(re.escape("[[ACTION]]"), action, task_description)

    for attempt in range(max_retries):
        model.add_user_turn(task_description)
        response = model.get_model_response()

        # Remove turns to save input space
        model.remove_turns(-1)

        # Parse conditions and effects
        conditions_effects = response.choices[0].message.content.strip()
        conditions, effects = _parse_conditions_effects(conditions_effects, agent, action)

        # Verification loop (TASK-007 integration)
        if flags.verification_loop and domain_knowledge is not None:
            # Build knowledge base from domain_knowledge for verification
            known_agents = set(domain_knowledge.get("agents", {}).keys())
            known_beliefs = set()
            for ag_name, ag_data in domain_knowledge.get("agents", {}).items():
                for kb_item in ag_data.get("knowledge_base", []):
                    # Extract belief/desire names
                    match = re.search(r'(BEL|DES)\([^,]+,\s*([^)]+)\)', kb_item)
                    if match:
                        known_beliefs.add(match.group(2).strip())

            # Verify conditions and effects
            result = verify_conditions_effects(
                conditions=conditions,
                effects=effects,
                known_agents=known_agents,
                known_beliefs=known_beliefs,
                agent_name=agent,
                action_name=action,
            )

            if not result.valid and attempt < max_retries - 1:
                # Generate feedback and retry
                feedback = format_errors_for_llm(result.errors)
                retry_prompt = format_regeneration_prompt(
                    original_prompt=task_description,
                    errors=result.errors,
                    original_response=conditions_effects,
                )
                task_description = retry_prompt
                print(f"      Verification failed, retrying ({attempt + 1}/{max_retries})...")
                continue

        # Return on success or after all retries
        return conditions, effects

    return conditions, effects


def get_occ_emotion(model, agent, action, domain_knowledge, task_description):
    task_remember = f"The beliefs and desires of agent {agent} are: "
    task_remember += "\n".join(domain_knowledge["agents"][agent]["knowledge_base"])

    task_remember = f"The actions of agent {agent} are: "
    task_remember += "\n".join(domain_knowledge["agents"][agent]["actions"])

    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    task_description = re.sub(re.escape("[[ACTION]]"), action, task_description)
    model.add_user_turn(task_remember + " " + task_description)
    response = model.get_model_response()

    #print(response.choices[0].message.content.strip())

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse occ emotion
    occ_emotion = response.choices[0].message.content.strip()
    occ_emotion = re.findall("\[\[.*?\]\]", occ_emotion)
    occ_emotion = [elem for elem in occ_emotion if elem != action and elem != agent]

    return occ_emotion


def get_action_mood(model, agent, action, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    task_description = re.sub(re.escape("[[ACTION]]"), action, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse mood
    mood = response.choices[0].message.content.strip()
    mood = re.findall("\[\[.*?\]\]", mood)
    mood = [re.sub("Mood\(.+?\)", "Mood(SELF)", elem) for elem in mood if elem != action and elem != agent]
    #final_mood = []
    """for i, elem in enumerate(mood):
        if re.findall("\[\[Mood\(SELF\).*?[<>=]+.+?\]\]", elem):
            final_mood.append(elem)
        elif re.findall("\[\[Mood\(.+?\).*?[<>=]+.+?\]\]", elem):
              final_mood.append(re.sub("Mood\(.+?\)", "Mood(SELF)", elem))"""

    return mood

def get_dialogue_tree(model, domain_knowledge, task_description):

    #task_remember = """
    #These are the actions agents can perform:
    #"""

    #for agent in domain_knowledge["agents"].keys():
    #    task_remember += f'\n{agent}:\n'
    #    for action in domain_knowledge["agents"][agent]["actions"].keys():
    #        task_remember += f'{action}:\n'

    #model.add_user_turn(task_remember + " " + task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    #model.remove_turns(-1)
    # model.add_model_turn(response)

    model.add_model_turn(response)

    # Parse dialogue tree
    dialogue_tree = response.choices[0].message.content.strip()
    dialogue_tree = re.findall("\[\[.*?\]\]", dialogue_tree)
    dialogue_tree = [re.sub(",[ ]+?\)]", "\)", elem) for elem in dialogue_tree] #if elem != agent]
    dialogue_tree = [re.sub("\[\[", "[[<", elem) for elem in dialogue_tree] #if elem != agent]
    dialogue_tree = [re.sub("\]\]", ">]]", elem) for elem in dialogue_tree] #if elem != agent]

    # print(response.choices[0].message.content.strip())

    return dialogue_tree

def get_dialogue_turns(model, domain_knowledge, agent, task_description):

    #task_remember = """
    #Dialogue turns are represented as [[CurrentState, NextState, Meaning, Style, UtteranceText]]
    #These are the dialogue turns agents can say:
    #"""
    #for dialogue in domain_knowledge["dialogue_tree"]:
    #    task_remember += f'\n{dialogue}:\n'

    task_description = re.sub("\[\[AGENT NAME\]\]", agent, task_description)

    model.add_user_turn(task_description)
    #model.add_user_turn(task_remember + " " + task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse speak actions
    speak_actions = response.choices[0].message.content.strip()
    speak_actions = re.findall("\[\[.*?\]\]", speak_actions)
    speak_actions = [re.sub(",[ ]+?\)]", "\)", elem) for elem in speak_actions if elem != agent]

    for i, sp in enumerate(speak_actions):
        sp = re.sub("\[\[|\]\]", "", sp)
        arguments = sp.split(",")
        if len(arguments) < 3:
            continue

        speak_action = "[[Speak(" + arguments[0] + "," + arguments[1] + "," + arguments[2] + "," + arguments[3] + ")]]"
        speak_actions[i] = speak_action

    # print(response.choices[0].message.content.strip())
    return speak_actions

def get_action_events(model, agent, action, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    task_description = re.sub(re.escape("[[ACTION]]"), action, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse events
    events = response.choices[0].message.content.strip()
    events = re.findall("\[\[.*?\]\]", events)
    events = [re.sub(",[ ]+?\)]", "\)", elem) for elem in events if elem != agent]

    return events

def get_agents(model, domain_knowledge):
    agents = get_agents_task(model, AGENT_TRANSLATION_TASK)
    # Add agents to domain knowledge
    domain_knowledge["agents"] = {agent: {} for agent in agents}
    print("Agents in the scenario: ", agents)
    return domain_knowledge

def get_agents_knowledge(model, domain_knowledge):
    for agent in tqdm(domain_knowledge["agents"].keys()):

        # Extract Beliefs and Desires
        bel_des_base = get_beliefs_desires(model, agent, BELIEFS_DESIRES_TRANSLATION_TASK)
        domain_knowledge["agents"][agent]["knowledge_base"] = bel_des_base

        # Extract Intentions conditioned on the Beliefs and Desires
        knowledge_base = f'The agent {agent} beliefs and desires are: ' + "\n".join(
            domain_knowledge["agents"][agent]["knowledge_base"])
        task_description = knowledge_base + "\n" + INTENTS_TRANSLATION_TASK
        intentions = get_intentions(model, agent, task_description)
        domain_knowledge["agents"][agent]["intentions"] = {intention : {} for intention in intentions}

    return domain_knowledge

def get_actions_plans(model, domain_knowledge):
    for agent in tqdm(domain_knowledge["agents"].keys()):
        print("Calculating action plan for agent ", agent)

        knowledge_base = f'The agent {agent} beliefs and desires are: ' + "\n".join(domain_knowledge["agents"][agent]["knowledge_base"])

        task_description = knowledge_base + "\n" + ACTION_PLAN_TRANSLATION_TASK

        for intention in tqdm(domain_knowledge["agents"][agent]["intentions"].keys()):
            # A plan is a sequence of action in the order they should occur
            # We assume the model outpus the sequence in the correct order when prompt to do so
            action_plan = get_agent_action_plan(model, agent, intention, task_description)
            domain_knowledge["agents"][agent]["intentions"][intention]["action_plan"] = action_plan
            domain_knowledge["agents"][agent]["actions"] = {action: {} for action in action_plan}

    return domain_knowledge

def get_agent_action_plan(model, agent, intention, task_description):

    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    task_description = re.sub(re.escape("[[INTENTION]]"), intention, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse events
    plan = response.choices[0].message.content.strip()
    plan = re.findall("\[\[.*?\]\]", plan)
    plan = [re.sub(",[ ]+?\)]", "\)", elem) for elem in plan if elem != agent]

    return plan


def get_actions_conditions_and_effects(model, domain_knowledge):
    flags = get_feature_flags()
    for agent in tqdm(domain_knowledge["agents"].keys()):
        print("Calculating action conditions and effects for agent ", agent)
        for intention in tqdm(domain_knowledge["agents"][agent]["intentions"].keys()):
            knowledge_base = f'The agent {agent} beliefs and desires are: ' + "\n".join(
                domain_knowledge["agents"][agent]["knowledge_base"])

            action_plan = f'These are the action the agent {agent} plans to to in this order to achieve the intention {intention}:\n' \
                          + "\n".join(domain_knowledge["agents"][agent]["intentions"][intention]["action_plan"])

            task_description = knowledge_base + "\n" + action_plan + "\n" + CONDITIONS_EFFECTS_TASK

            for action in domain_knowledge["agents"][agent]["intentions"][intention]["action_plan"]:
                # Pass domain_knowledge to enable verification loop (TASK-007)
                conditions, effects = get_action_conditions_effects(
                    model, agent, action, task_description,
                    domain_knowledge=domain_knowledge if flags.verification_loop else None
                )
                domain_knowledge["agents"][agent]["actions"][action] = {"conditions": conditions, "effects": effects}

    return domain_knowledge



def get_emotional_state(model, domain_knowledge):
    for agent in tqdm(domain_knowledge["agents"].keys()):
        print("Calculating emotional state for agent ", agent)

        # What was the initial emotion of the agent at the beggining of the scenario
        occ_emotion = get_initial_occ_emotion(model, agent, INITIAL_EMO_TASK)
        domain_knowledge["agents"][agent]["initial_occ_emotion"] = occ_emotion

        # What is the initial mood of the agent at the beggining of the scenario
        #initial_mood = get_initial_mood(model, agent, action, INITIAL_MOOD_TASK)
        #domain_knowledge["agents"][agent]["initial_mood"] = initial_mood

        for action in domain_knowledge["agents"][agent]["actions"].keys():
            # What emotion did the agent felt after performing the action
            occ_emotion = get_occ_emotion(model, agent, action, domain_knowledge, ACTIONS_EMO_APPRAISAL)
            domain_knowledge["agents"][agent]["actions"][action]["occ_emotion"] = occ_emotion

            # What emotion is required to perform the action
            occ_emotion = get_occ_emotion(model, agent, action, domain_knowledge, EMOTION_CONDITION_TASK)
            domain_knowledge["agents"][agent]["actions"][action]["emotion_condition"] = occ_emotion

            # What mood did the agent felt after performing the action
            occ_emotion = get_occ_emotion(model, agent, action, domain_knowledge, ACTIONS_EMO_APPRAISAL)
            domain_knowledge["agents"][agent]["actions"][action]["occ_emotion"] = occ_emotion

            # What mood is required to perform the action
            #action_mood = get_action_mood(model, agent, action, ACTION_MOOD)
            #domain_knowledge["agents"][agent]["actions"][action]["emotion_condition"] = occ_emotion

    return domain_knowledge

def get_initial_occ_emotion(model, agent, task_description):
    task_description = re.sub(re.escape("[[AGENT NAME]]"), agent, task_description)
    model.add_user_turn(task_description)
    response = model.get_model_response()

    # Remove turns to save input space
    model.remove_turns(-1)
    # model.add_model_turn(response)

    # Parse events
    plan = response.choices[0].message.content.strip()
    plan = re.findall("\[\[.*?\]\]", plan)
    plan = [re.sub(",[ ]+?\)]", "\)", elem) for elem in plan if elem != agent]

    return plan

def generate_dialogue_states(model, domain_knowledge):
    dialogue_tree = get_dialogue_tree(model, domain_knowledge, DIALOGUE_TREE_TASK)
    domain_knowledge["dialogue_tree"] = dialogue_tree
    return domain_knowledge

def generate_speak_actions(model, domain_knowledge):
    dialogue_tree = f'The dialogue turns available are:\n' + "\n".join(domain_knowledge["dialogue_tree"])

    # Speak actions
    for agent in tqdm(domain_knowledge["agents"].keys()):
        speak_actions = get_dialogue_turns(model, domain_knowledge, agent, SPEAK_ACTION_TASK)
        domain_knowledge["agents"][agent]["speak_actions"] = {speak_action: {} for speak_action in speak_actions}

    return domain_knowledge

def get_speak_actions_conditions_and_effects(model, domain_knowledge):
    for agent in tqdm(domain_knowledge["agents"].keys()):
        print("Calculating speak action conditions and effects for agent ", agent)

        knowledge_base = f'The agent {agent} beliefs and desires are: ' + "\n".join(
            domain_knowledge["agents"][agent]["knowledge_base"]) + "\n The agent's Intentions are:\n" \
                         + "\n".join(domain_knowledge["agents"][agent]["intentions"].keys())

        dialogue_tree = f"The dialogue state machine of the scenario is: " + "\n".join(domain_knowledge["dialogue_tree"])

        task_description = knowledge_base + dialogue_tree + SPEAK_CONDITIONS_EFFECTS

        for speak_action in tqdm(domain_knowledge["agents"][agent]["speak_actions"].keys()):
                conditions, effects = get_action_conditions_effects(model, agent, speak_action, task_description)
                domain_knowledge["agents"][agent]["speak_actions"][speak_action] = {"conditions": conditions, "effects": effects}

    return domain_knowledge

def get_speak_emotional_state(model, domain_knowledge):
    for agent in tqdm(domain_knowledge["agents"].keys()):
        print("Calculating speak emotional state for agent ", agent)

        for action in tqdm(domain_knowledge["agents"][agent]["speak_actions"].keys()):
            # What emotion did the agent felt after performing the action
            occ_emotion = get_occ_emotion(model, agent, action, domain_knowledge, ACTIONS_EMO_APPRAISAL)
            domain_knowledge["agents"][agent]["speak_actions"][action]["occ_emotion"] = occ_emotion

            # What emotion is required to perform the action
            occ_emotion = get_occ_emotion(model, agent, action, domain_knowledge, EMOTION_CONDITION_TASK)
            domain_knowledge["agents"][agent]["speak_actions"][action]["emotion_condition"] = occ_emotion

            # What mood did the agent felt after performing the action
            #occ_emotion = get_occ_emotion(model, agent, action, ACTIONS_EMO_APPRAISAL)
            #domain_knowledge["agents"][agent]["actions"][action]["occ_emotion"] = occ_emotion

            # What mood is required to perform the action
            #action_mood = get_action_mood(model, agent, action, ACTION_MOOD)
            #domain_knowledge["agents"][agent]["actions"][action]["emotion_condition"] = occ_emotion

    return domain_knowledge


def domainknowledge_to_json(domain_knowledge, scenario_name):
    # Convert to json so that I can see
    result = json.dumps(domain_knowledge, indent=4)
    result = re.sub("\[\[|\]\]", "", result)
    with open(f"Data/{re.sub(' ', '_', scenario_name)}.json", "w") as outfile:
        outfile.write(result)


def generate_scenario(scenario_name, scenario_description):
    if os.path.exists(f"Data/{re.sub(' ', '_', scenario_name)}.json"):
        """input("There's already a scenariowith that name, want to continue it?: ")
        if input = """
        with open(f"Data/{re.sub(' ', '_', scenario_name)}.json") as file:
            domain_knowledge = json.load(file)
            continue_domain_knowledge = True

    else:
        # Object at the end that will be converted to json
        # Dict of agents, which agent is a dict with knowledge base and actions
        # Knowledge base has a list of desires and beliefs
        # Actions is a dict with actions as keys, and have conditions, effects, occ_emotion and mood
        domain_knowledge = {"scenario_name": scenario_name, "scenario_description": scenario_description}
        continue_domain_knowledge = False

    print("Initializing model...")
    # Use feature flags to determine model (TASK-005 integration)
    flags = get_feature_flags()

    # Record feature flags in domain_knowledge for experiment tracking
    if not continue_domain_knowledge:
        domain_knowledge["feature_flags"] = flags.to_dict()
        domain_knowledge["model"] = "gpt-4o" if flags.use_gpt4 else "gpt-3.5-turbo"

    if flags.use_gpt4:
        model = get_model(use_gpt4=True)
        print(f"  Using GPT-4 model: {model.model_id}")
    else:
        model = OpenAIHandler(api_key=API_KEY, model_id='gpt-3.5-turbo')
        print(f"  Using baseline model: gpt-3.5-turbo")
    print(f"  Active features: {flags.enabled_features() or 'None (baseline)'}")

    # Initialize prompt manager (TASK-009 integration)
    prompt_manager = get_prompt_manager(use_enhanced=flags.cot_enhancement)
    if flags.cot_enhancement:
        print("  Using enhanced Chain-of-Thought prompts")

    # Explain task to model
    describe_task(model, TASK_DESCRIPTION)

    print("Starting Scenario generation and translation ... ")
    # Use generative power of model to give more detail to the scenario, because the prompt is very short
    scenario_task(model, scenario_description, SCENARIO_DESCRIPTION_GENERATIVE_TASK)
    if not continue_domain_knowledge:
        # domain_knowledge["extended_scenario_description"] = scenario_description
        domain_knowledge["last_ended"] = "scenario"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "scenario":
        # Extrapolate the agents in the scenario
        domain_knowledge = get_agents(model, domain_knowledge)
        domain_knowledge["last_ended"] = "agents"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "agents":
        # Extrapolate each agent beliefs and desires and intents
        # The fuction updates the object domain_knowledge with each agents beliefs, desires and intentions
        # Intentions are conditioned bu the beliefs and desires of an agent, as per the BDI architecture
        domain_knowledge = get_agents_knowledge(model, domain_knowledge)
        domain_knowledge["last_ended"] = "knowledge"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "knowledge":
        # Extrapolate plausible action plans for each agent to achieve their intentions
        domain_knowledge = get_actions_plans(model, domain_knowledge)
        domain_knowledge["last_ended"] = "actions_plans"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "actions_plans":
        # Extrapolate the conditions and effects of each action
        domain_knowledge = get_actions_conditions_and_effects(model, domain_knowledge)
        domain_knowledge["last_ended"] = "conditions_effects"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "conditions_effects":
        # Emotionally appraise each action - How the agent emotionally reacts to the action after performing it?
        # Compute the mood after performing each action
        # Compute initial mood
        domain_knowledge = get_emotional_state(model, domain_knowledge)
        domain_knowledge["last_ended"] = "emotional_state"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "emotional_state":
        # Generate Dialogue State Machine
        domain_knowledge = generate_dialogue_states(model, domain_knowledge)
        domain_knowledge["last_ended"] = "dialogues"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "dialogues":
        # Decide which characters can say what by attributing speak actions to each of them
        domain_knowledge = generate_speak_actions(model, domain_knowledge)
        domain_knowledge["last_ended"] = "speak_actions"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "speak_actions":
        # Extrapolate speak actions conditions and effects
        domain_knowledge = get_speak_actions_conditions_and_effects(model, domain_knowledge)
        domain_knowledge["last_ended"] = "speak_conditions_effects"
        domainknowledge_to_json(domain_knowledge, scenario_name)

    if not continue_domain_knowledge or domain_knowledge["last_ended"] == "speak_conditions_effects":
        # Emotionally appraise each action - How the agent emotionally reacts to the speak action after performing it?
        domain_knowledge = get_speak_emotional_state(model, domain_knowledge)
        domain_knowledge["last_ended"] = "end"
        ##########################################################################################

    # Record usage statistics if using ModelHandler (TASK-005)
    if hasattr(model, 'get_usage_summary'):
        domain_knowledge["usage_stats"] = model.get_usage_summary()

    domainknowledge_to_json(domain_knowledge, scenario_name)

    return domain_knowledge


if __name__ == "__main__":
    scenario_name = input("Write the scenario name: ")
    scenario_description = input("Write a small description of a social scenario: ")
    start_time = time.time()

    generate_scenario(scenario_name, scenario_description)

    ###########################################################################################
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))