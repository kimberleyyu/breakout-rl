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
        agent = qlearningagent.QLearner(epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0)
        action = QLearner.getAction(state)
        nextState, reward, done, info = env.step(action)
        QLearner.update(state, action, nextState, reward)
        nextState = state
        # end if done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# env.monitor.close()