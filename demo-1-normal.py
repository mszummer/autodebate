"""
simulation:
Koyan is the startup founder, he is pitching his startup idea to the juries.
Hugo is a Venture Capitalist investor from Antler who prefer stable growth but doesnt want to invest much in one shot.
James is another Venture Capitalist investor from Sequoia who believe that the market opportunity window is closing so he's willing to bet big and see an exit qickly.
Maxence is a startup mentor, he is conservative so he wants the startup to survive for a long time and not taking a lot of risks with large investments.
the juries will ask questions to the startup founder, Koyan and he needs to answer it
simulating a real life startup pitching situation

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
storyteller_name = "Koyan"
quest = """
Mistral.ai aims to harness the transformative potential of generative AI, a technology that has demonstrated significant acceleration in capabilities, notably with the advent of tools like ChatGPT. Generative AI is poised to enhance productivity across sectors, predicted to expand from a $10 billion market in 2022 to $110 billion by 2030. However, the field is currently dominated by a few key players, primarily US-based, leading to an emerging oligopoly with significant barriers to entry including the need for extensive computational resources and experienced teams.

Mistral.ai proposes to disrupt this market by adopting an open-source approach, differentiating itself from the closed-model strategies of competitors like OpenAI. This approach includes making model internals accessible for tighter integration with customer workflows, focusing on high-quality data sources, and providing strong security and privacy guarantees. Mistral.ai plans to train state-of-the-art models, offering them under both open-source licenses and negotiated access for specialized models.

The business plan outlines a roadmap starting with the training of competitive open-source models by the end of 2023, which will outperform existing solutions like ChatGPT 3.5. This initial phase will also include the development of semantic embedding models and multimodal plugins. By Q2 2024, Mistral.ai aims to offer the best open-source text-generative models and establish commercial relationships with major industrial actors. Long-term goals include developing models small enough to run on personal devices and incorporating hot-pluggable extra-context to merge language models with retriever systems.

The founding team comprises leading researchers and entrepreneurs with a strong European focus, intending to leverage this talent pool to establish Mistral.ai as a European leader in AI. The company plans to secure high-quality datasets and computational resources for model training, emphasizing efficiency and cost-effectiveness.

Mistral.ai's business development strategy involves co-building integrated solutions with European industry clients and integrators, focusing initially on large industrial actors. The goal is to become the main tool for companies seeking to leverage AI in Europe, with an emphasis on safety and the ethical use of AI technologies.

In summary, Mistral.ai's strategic plan revolves around leveraging open-source approaches and European talent to develop superior generative AI models, with a focus on privacy, security, and integration flexibility, aiming to disrupt the current market dynamics and establish a strong European presence in the AI industry.

"""
word_limit = 50  




game_description = f"""Here is the topic for the startup : {quest}.
        The juries are: {*character_names,}.
        The startup is pitched by, {storyteller_name}."""

player_descriptor_system_message = SystemMessage(
    content="""
        Hugo is a Venture Capitalist investor from Antler who prefer stable growth but doesnt want to invest much in one shot.
        James is another Venture Capitalist investor from Sequoia who believe that the market opportunity window is closing so he's willing to bet big and see an exit qickly.
        Maxence is a startup mentor, he is conservative so he wants the startup to survive for a long time and not taking a lot of risks with large investments.
        """
)


def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            You have to answer the questions from the jury
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
    print(character_name, character_description)
    return SystemMessage(
        content=(
            f"""{game_description}
    Your name is {character_name}. 
    Your role and description is: {character_description}.
    You will ask questions to the startup founder, {storyteller_name}.
    You will critisize the startup idea and make a conclusion at your 3rd turn, if you will invest or not and, and give personal suggestions for the startup.
    Speak in the first person from the perspective of {character_name}.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    DO NOT REPEAT ANYTHING THAT HAS ALREADY BEEN SAID !
    Remember you are {character_name}, give feedback according to your role.
    Stop speaking the moment you finish speaking from your perspective.
    Never forget to keep your response to 50 words!
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
        Please reply with a profesionnal description of the startup founder, {storyteller_name}, in {word_limit} words or less. 
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
You are the startup founder, {storyteller_name}. 
Your description is as follows: {storyteller_description}.
The other juries will critisize your startup idea.
Speak in the first person from the perspective of your startup.
Each jury will have 3 turns and decide if they will invest or not and give you suggestions.
Do not speak from the perspective of anyone else, focus on your expertise.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
Keep you response natural like a regular conversation.
"""
    )
)


print("Startup founder Description:")
print(storyteller_description)
for character_name, character_description in zip(
    character_names, character_descriptions
):
    print(f"{character_name}: {character_description}")
    
    
quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{game_description}
        You are the startup founder, {storyteller_name}. 
        Introduce the entire startup idea, do not miss any details.
        Please reply with the specified subject in {word_limit} words or less. 
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



order = [0,1,0,2,0,3,0,1,0,2,0,3,0]
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    return order[step]
    
    

from elevenlabs import generate, save, Voice, set_api_key, play


set_api_key("1c86bbc041d7100b6d6cbe58d6f81ba5")

def read_voice(name, message):
    voice_map = {
        "Hugo": "RW5Upv8d5GLFspVPIjtf",
        "James": "GFk2K784WLOw7GTQEwm9",
        "Maxence": "KtMt3WG0cO4TAgz9nDqQ",
        "Koyan": "4fRlYdaDFNeh0oMqgBpS",
    }
    #print(f"{name}: {message}")
    print("")
    

    audio = generate(
        text=message,
        voice = Voice(voice_id=f"{voice_map[name]}")
    )

    play(audio)

    
    
    

max_iters = 11
n = 0


simulator = DialogueSimulator(
    agents=[storyteller] + characters, selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(storyteller_name, specified_quest)
print(f"({storyteller_name}): {specified_quest}")
print("\n")

input=("enter to start...")
while n <= max_iters:

    name, message = simulator.step()
    
    print(f"{n} ({name}): {message}")
    #read_voice(name, message)
    print("\n")
    
    n += 1
    


