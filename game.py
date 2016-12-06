import gym
import qlearningagent
env = gym.make('Breakout-v0')
# for monitoring how we are doing
# env.monitor.start('/tmp/breakout-experiment-1')

# play the game 5 times
for i_episode in range(5):
    state = env.reset()
    for t in range(100):
        env.render()
        legalActions = range(env.action_space.n)
        print env.observation_space

        agent = qlearningagent.QLearner(legalActions, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0) 
        action = agent.getAction(state)
        nextState, reward, done, info = env.step(action)
        agent.update(state, action, nextState, reward)
        nextState = state
        # end if done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# env.monitor.close()
