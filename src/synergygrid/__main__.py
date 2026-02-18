from synergygrid import Agent, AgentAction

import random

def main():
    agent = Agent()
    agent.render()

    for i in range(25):
        rand_action = random.choice(list(AgentAction))
        print(rand_action)

        agent.perform_action(rand_action)
        agent.render()

if __name__ == "__main__":
    main()