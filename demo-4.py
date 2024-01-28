""" 
try to do a synthetic data generation lab in form of debate conversation
the objectives is to make a llm smarter and learn about a topic he did not know about before
we have to introduce a user input, and make the user learn 


board member act as a teacher checking if the student agent understood the topic and learned something new
the student agent is the one who is learning and asking questions
the other agents are the one who are answering the questions

agents:
Teacher 1,2,3: gpt 4
Student: gpt 3.5
Supervisor: gpt 4

we want gpt 3.5 to learn to solve the following problem:
Question: Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.
Answer: 4
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
    
    

character_names = ["Teacher 1", "Teacher 2", "Teacher 3"]
external_agent = "Student"
storyteller_name = "Supervisor"
quest = """
learn to solve: -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r
"""
word_limit = 50  


student_agent_message = SystemMessage(
    content=f"""
    You are the student, {external_agent}.
    Your role is to learn to solve the following problem: {quest}.
    Each time, solve the problem and give your answers and steps to the student.
    You will be given advice and ask the teachers for directions.
    You are not allowed to ask directky for the answer and you won't be given the answer directly.
    You're limited to 100 words per response.
    """)
    

game_description = f""".
        The participants are: {*character_names,}.
        {storyteller_name} is a supervisor making sure the student is learning.
        {external_agent} is the student who is learning to solve the problem.
        We want the student to learn to solve the following problem: {quest}.
        the answer of the problem is EXACTLY 4 but never give the answer directly to the student.
        You're limited to 100 words per response.
        """

player_descriptor_system_message = SystemMessage(
    content="""
        Teacher 1, Teacher 2 and Teacher 3 are mathematicians specialized in equation solving.
        their role is to give advice and directions to make sure the student understand how to solve the problem
        You're limited to 100 words per response.
        """
)


def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            You have help the student solve the following problem and explain it: {quest}.
            Please reply with a professional and concise description for each  {character_name} given your personas, in {word_limit} words or less. 
            Speak directly to {character_name}.
            Do not add anything else.
            You're limited to 100 words per response."""
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
    Do not give the answer directly to the student, but give him advice and directions to solve the problem if he's struggling.
    Do not give him any numbers.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    Remember you are {character_name}, give feedback according to your role.
    Stop speaking the moment you finish speaking from your perspective.
    Keep you response natural like a regular conversation.
    Do not add anything else.
    You're limited to 100 words per response.
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
        Please reply with a profesionnal description of the supervisor {storyteller_name}, in {word_limit} words or less. 
        Your role is to make sure the student is actively learning by giving directions and asking questions.
        You are also in charge of checking the student answer, and if its correct, congrat the student and end the convo.
        Do not speak from the perspective of anyone else.
        Keep the conversation natural.
        Speak directly to {storyteller_name}.
        Never forget to keep your response to less than 100 words!
        Do not add anything else.
        You're limited to 100 words per response.
        """
    ),
]
storyteller_description = ChatOpenAI(temperature=1.0, model='gpt-4')(
    storyteller_specifier_prompt
).content

storyteller_system_message = SystemMessage(
    content=(
        f"""{game_description}
You are the supervisor, {storyteller_name}. 
Your description is as follows: {storyteller_description}.
Keep the conversation natural.
Never forget to keep your response to less than 100 words!
Stop speaking the moment you finish speaking from your perspective.
Do not add anything else.
You're limited to 100 words per response.
"""
    )
)


    
quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{game_description}
        Introduce the entire debate, do not add anything else.
        Start by asking the student to solve the following problem: {quest}.
        Please reply with the specified subject in less than 100 words 
        Do not add anything else.
        You're limited to 100 words per response.
        """
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



steps_round = [1,0,2,1,0,3,1,0,4,1]
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    return steps_round[step % len(steps_round)]

from elevenlabs import generate, Voice, set_api_key, play


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

max_iters = 50
n = 0


# initiate the student agent
student_agent = DialogueAgent(
    name=external_agent,
    system_message=student_agent_message,
    model=ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo"),
)

simulator = DialogueSimulator(
    agents=[storyteller] +[student_agent]+ characters, selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(storyteller_name, specified_quest)

while n <= max_iters:

    name, message = simulator.step()
    
    print(f"{n} ({name}): {message}")
    #read_voice(name, message)
    print("\n")
    
    n += 1
    


