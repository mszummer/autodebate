"""

Debate Topic: "Intersecting Paths: Quantum Technology, Historical Lessons, and Our Future"
try to make the student learn
LLM "Quantum" (Quantum Physics Expert):
- Description: Specializes in quantum mechanics, theoretical physics, and advanced scientific concepts. Able to explain complex theories and recent discoveries.
- Role: Discusses the intricacies of quantum physics, its philosophical implications, and its potential future applications.

LLM "Historia" (History and Culture Scholar):
- Description: Expert in world history, cultural evolution, and historical analysis. Skilled in drawing connections between past events and current global situations.
- Role: Provides historical context to discussions, highlights how history shapes current events, and offers insights into cultural dynamics.

LLM "Futurist" (Technology and Future Trends Analyst):
- Description: Focuses on emerging technologies, future societal trends, and speculative scenarios. Engages with ideas about how technology will shape the future.
- Role: Projects potential future developments based on current technological trends and explores their societal and ethical implications.
"""


from typing import Callable, List
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

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
    
    

character_names = ["Quantum", "Historia", "Futurist"]
storyteller_name = "Student"
quest = """
Debate Topic: "Intersecting Paths: Quantum Technology, Historical Lessons, and Our Future"
"""
word_limit = 150  




game_description = f"""Here is the topic for the debate : {quest}.
        The participants are: {*character_names,}.
        {storyteller_name} is a student who wants to learn more. he will ask questions to the participants and try to understand the topic deeper.
        all the response should NOT exceed 100 words.
        """

player_descriptor_system_message = SystemMessage(
    content="""
        Quantum is in quantum mechanics, theoretical physics, and advanced scientific concepts, and he is able to explain complex theories and recent discoveries.
        his role is to discusse the intricacies of quantum physics, its philosophical implications, and its potential future applications.

        Historia is an Expert in world history, cultural evolution, and historical analysis, and he is skilled in drawing connections between past events and current global situations.
        his role is to provide historical context to discussions, highlights how history shapes current events, and offers insights into cultural dynamics.
        
        Futurist is focused on emerging technologies, future societal trends, and speculative scenarios, and he engages with ideas about how technology will shape the future.
        his role is to examine projects potential future developments based on current technological trends and explores their societal and ethical implications.
        """
)


def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            You have to debate on the topic: {quest}.
            Please reply with a professional and concise description for each  {character_name} given your personas, in {word_limit} words or less. 
            Speak directly to {character_name}.
            Do not add anything else."""
        ),
    ]
    character_description = ChatOpenAI(temperature=1.0, model="gpt-4")(
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
        Please reply with a profesionnal description of the student {storyteller_name}, in {word_limit} words or less. 
        Your role is to ask questions and try to learn more about the topic from a student perspective.
        the talking order is: student ,quantum, student, historia, student, futurist, student, quantum, historia, futurist, student.
        Ask questions wisely to the participants in order to learn more.
        At the end say what you learned.
        Do not speak from the perspective of anyone else.
        Keep the conversation natural.
        Speak directly to {storyteller_name}.
        Never forget to keep your response to less than 100 words!
        Do not add anything else.
        """
    ),
]
storyteller_description = ChatOpenAI(temperature=1.0, model='gpt-4')(
    storyteller_specifier_prompt
).content

storyteller_system_message = SystemMessage(
    content=(
        f"""{game_description}
You are the student, {storyteller_name}. 
Your description is as follows: {storyteller_description}.
the talking order is: 
student ,quantum, student,historia, student, futurist,student ,quantum, student,historia, student, futurist,student ,quantum, student,historia, student, futurist,student.
So choose your questions wisely.
Keep the conversation natural.
At the last round (n=19, when it's the 10th time your talking, after futurist),you have to make a conclusion about what you learned.
Stick to your role, never predict response.
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
        Introduce the entire debate, do not add anything else.
        Start by asking a question to Quantum
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
    model=ChatOpenAI(temperature=1.0, model="gpt-4"),
)



def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    """
    If the step is even, then select the storyteller
    Otherwise, select the other characters in a round-robin fashion.

    For example, with three characters with indices: 1 2 3
    The storyteller is index 0.
    Then the selected index will be as follows:

    step: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 

    idx:  0  1  0  2  0  3  0  1  0  2  0  3  0  1  0  2  0  3  0 
    """
    if step % 2 == 0:
        idx = 0
    else:
        idx = (step // 2) % (len(agents) - 1) + 1
    
    return idx

from elevenlabs import generate, save, Voice, set_api_key, play


set_api_key("1c86bbc041d7100b6d6cbe58d6f81ba5")

def read_voice(name, message):
    voice_map = {
        "Quantum": "RW5Upv8d5GLFspVPIjtf",
        "Historia": "GFk2K784WLOw7GTQEwm9",
        "Futurist": "KtMt3WG0cO4TAgz9nDqQ",
        "Student": "4fRlYdaDFNeh0oMqgBpS",
    }
    #print(f"{name}: {message}")
    print("")
    

    audio = generate(
        text=message,
        voice = Voice(voice_id=f"{voice_map[name]}")
    )

    play(audio)

    
    
    
# print character descriptions
for character_name, character_description in zip(
    character_names, character_descriptions
):
    print(f"{character_name}:\n{character_description}\n")
    
    
# print student description
print(f"{storyteller_name}:\n{storyteller_description}\n")

max_iters = 19
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
    


