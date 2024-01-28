"""

Debate on Implementing a New Green Technology in Urban Areas:

Scenario: A city is considering implementing a new green technology (like solar-powered public transport).
Eco's Stance: Advocates for the technology due to its environmental benefits.
Techno's Stance: Discusses the tech aspects, potential challenges, and future integration with other smart city initiatives.
Econo's Stance: Analyzes the cost, potential economic benefits, and financial feasibility of implementing this technology.
Modera's Role: Guides the debate to cover all aspects (environmental, technological, economic), ensuring a balanced discussion.


"""


from typing import Callable, List
from dotenv import load_dotenv
load_dotenv()

from rich import print

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")
        
        

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message
    
    

character_names = ["Hugo", "James", "Maxence"]
storyteller_name = "Moderator"
quest = """
Debate on Implementing a New Green Technology in Urban Areas:
Scenario: A city is considering implementing a new green technology (like solar-powered public transport).
"""
word_limit = 150  




game_description = f"""Here is the topic for the debate : {quest}.
        The participants are: {*character_names,}.
        The debate is moderated by, {storyteller_name}.
        the moderator is designed to be neutral, ensuring a balanced and fair debate. Keeps the discussion on track, asks probing questions, and summarizes key points."""

player_descriptor_system_message = SystemMessage(
    content="""
        Hugo is specialized in environmental issues, sustainability, and climate change. Advocates for eco-friendly policies and practices..
        his role is to bring in facts about environmental impact, argues for sustainable solutions, and emphasizes the long-term benefits of eco-conscious decisions.
        
        James is focused on technological advancements, innovation, and the impact of tech on society. Enthusiastic about AI, IoT, and emerging tech trends.
        his role is to highlights how technology can provide solutions, discusses the role of innovation in solving current problems, and examines the future of tech integration.
        
        Maxence is an expert in economics, finance, and business. Analyzes the economic implications and viability of decisions and policies.
        his role is to examine the financial aspects, market impacts, and economic feasibility of ideas and solutions proposed in the debate.
        """
)


def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            You have to debate on the topic: {quest}.
            Please reply with a professional and concise description for each  {character_name} given his focus and role, in {word_limit} words or less. 
            Speak directly to {character_name}.
            Do not add anything else."""
        ),
    ]
    character_description = ChatOpenAI(temperature=1.0)(
        character_specifier_prompt
    ).content
    return character_description


def generate_character_system_message(character_name, character_description):
    return SystemMessage(
        content=(
            f"""{game_description}
    Your name is {character_name}. 
    Your role and description is: {character_description}.
    Speak in the first person from the perspective of {character_name}.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    DO NOT REPEAT ANYTHING THAT HAS ALREADY BEEN SAID !
    Remember you are {character_name}, give feedback according to your role.
    Never forget to keep your response less than 100 words!
    Stop speaking the moment you finish speaking from your perspective.
    Keep you response natural like a regular conversation.
    Do not add anything else.
    """
        )
    )


character_descriptions = [
    
    generate_character_description(character_name) for character_name in character_names
]

character_system_messages = [
    generate_character_system_message(character_name, character_description)
    for character_name, character_description in zip(
        character_names, character_descriptions
    )
]

storyteller_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Please reply with a profesionnal description of the debate moderator {storyteller_name}, in {word_limit} words or less. 
        Speak directly to {storyteller_name}.
        Do not add anything else.
        """
    ),
]
storyteller_description = ChatOpenAI(temperature=1.0)(
    storyteller_specifier_prompt
).content

storyteller_system_message = SystemMessage(
    content=(
        f"""{game_description}
You are the debate moderator, {storyteller_name}. 
Your description is as follows: {storyteller_description}.
Do not speak from the perspective of anyone else, focus on your expertise.
Never forget to keep your response to less than 100 words!
Stop speaking the moment you finish speaking from your perspective.
Do not add anything else.
"""
    )
)


    
quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{game_description}
        You are the debate moderator, {storyteller_name}. 
        Introduce the entire debate, do not add anything else.
        Please reply with the specified subject in less than 100 words 
        Do not add anything else."""
    ),
]
specified_quest = ChatOpenAI(temperature=1.0)(quest_specifier_prompt).content

print(f"Original topic:\n{quest}\n")
print(f"Detailed topic:\n{specified_quest}\n")





characters = []

for character_name, character_system_message in zip(
    character_names, character_system_messages
):
    characters.append(
        DialogueAgent(
            name=character_name,
            system_message=character_system_message,
            model=ChatOpenAI(temperature=1.0, model="gpt-4"),
        )
    )


    
storyteller = DialogueAgent(
    name=storyteller_name,
    system_message=storyteller_system_message,
    model=ChatOpenAI(temperature=0.1),
)



def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    if step % 2 == 0:
        idx = 0
    else:
        idx = (step // 2) % (len(agents) - 1) + 1
    
    return idx

from elevenlabs import generate, save, Voice, set_api_key, play


set_api_key("1c86bbc041d7100b6d6cbe58d6f81ba5")

def read_voice(name, message):
    voice_map = {
        "Hugo": "RW5Upv8d5GLFspVPIjtf",
        "James": "GFk2K784WLOw7GTQEwm9",
        "Maxence": "KtMt3WG0cO4TAgz9nDqQ",
        "Moderator": "4fRlYdaDFNeh0oMqgBpS",
    }
    #print(f"{name}: {message}")
    print("")
    

    audio = generate(
        text=message,
        voice = Voice(voice_id=f"{voice_map[name]}")
    )

    play(audio)

    
    
    

max_iters = 12
n = 0


simulator = DialogueSimulator(
    agents=[storyteller] + characters, selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(storyteller_name, specified_quest)

while n <= max_iters:

    name, message = simulator.step()
    
    print(f"{n} ({name}): {message}")
    #read_voice(name, message)
    print("\n")
    
    n += 1
    


