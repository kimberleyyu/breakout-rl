import gym
import qlearningagent
env = gym.make('Breakout-v0')
# for monitoring how we are doing
# env.monitor.start('breakout-experiment-2')
legalActions = range(env.action_space.n)
agent = qlearningagent.QLearner(legalActions, featureVersion = 1, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0)
# play the game 5 times
for i_episode in range(1000):
    state = env.reset()  
    for t in range(10000):
        env.render()
        action = agent.getAction(state)
        nextState, reward, done, info = env.step(action)
        print reward
        agent.update(state, action, nextState, reward)
        nextState = state
        # end if done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# env.monitor.close()
