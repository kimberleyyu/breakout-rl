import gym
import qlearningagent
import numpy as np
import matplotlib.pyplot as plt
import argparse

TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
PADDLE_LEN = 16

def ballFell(state):
    for i in range(SCREEN_L, SCREEN_R):
        if state[TOP_PADDLE_ROW][i][0] == 0 and state[TOP_PADDLE_ROW][i+1][0] != 0 and state[TOP_PADDLE_ROW][i+3][0] == 0:
            return True
    return False

## input command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--'featureVersion = 2, epsilon=0.05,gamma=0.99,alpha=0.02, numTraining)

parser.add_argument('--path', default='.', required=False, type=str, help='Path to folder that stores all inputs and outputs. Default: current folder')
parser.add_argument('--event_type', default='NA', required=False, type=str, help='SE, RI, AFE, ALE, or TandemUTR. Default: NA')
parser.add_argument('--event_info', required=False, type=str, help='to get strands of events. Must contain columns "event" and "strand" (full path)')
parser.add_argument('--motif_type', default='kmer', required=False, type=str, help='kmer or pwm')
parser.add_argument('--motif', required=True, type=str, help='motif to test, either a kmer string or filename of PWM table')
parser.add_argument('--genome', required=False, type=str, help='reference genome fasta file for getting sequences around SNPs (full path)')
#    parser.add_argument('--input_path', required=True, type=str, help='Foreground SNP table to test for enrichment against background SNPs. Must contain columns "snp_position", "snp_id", and "snp_chr"')
# the actual output with snp, event, statistic, FDR, pvalue, etc instead of position (get positions from background file instead)
parser.add_argument('--input_path', required=True, type=str, help='annotated matrixQTL output file. Must contain columns "snps", "event", "pvalue", and "FDR"')
parser.add_argument('--background_path', default='genotype.code.02.10.16.dat', required=True, type=str, help='Background SNP table. Must contain columns "snp_position", "snp_id", "snp_chr", "genotype_code_0", and "genotype_code_2"')
args = parser.parse_args()

path = args.path


env = gym.make('Breakout-v0')
# for monitoring how we are doing
# env.monitor.start('breakout-experiment-3')

### implement systems io stuff (actually athma's parser stuff) for the arguments in
#QLearner init (epsilon, gamma, alpha, numTraining)
#fig names to save
#Qlearner vs. Qlearner+ (+ ver has prev state incorporated to calculate better features, like trajectory)

legalActions = range(env.action_space.n)
agent = qlearningagent.QLearner(legalActions, featureVersion = 2, epsilon=0.05,gamma=0.99,alpha=0.02, numTraining=10000)
# agent = qlearningagent.QLearnerPlus(legalActions, featureVersion = 2, epsilon=0.05,gamma=0.99,alpha=0.02, numTraining=10000)
lengths = []
rewards = []
weights_over_time = []
end = False
# play the game 5 times
for i_episode in range(5050):
    r = 0
    state = env.reset()
    #prev_state = state
    for t in range(10000):
        env.render()
        action = agent.getAction(state)
        # action = agent.getAction(state, prev_state)
        nextState, reward, done, info = env.step(action)
        r +=reward
        if ballFell(nextState):
            reward = -1
        weights = agent.update(state, action, nextState, reward)
        # weights = agent.update(state, prev_state, action, nextState, reward)
        weights_over_time.append([weights['paddlex'],weights['ballx'], weights['bally']])
        if abs(weights["ballx"]) > 10**305:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            end = True
            break
        # nextState = state
        prev_state = state
        state = nextState
        # end if done
        if done:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            break
    if end:
        break
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lengths)
fig.savefig('lengthsdis.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(rewards)
fig.savefig('rewardsdis.png')

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(weights_over_time)
fig.savefig('weightsnondisz.png')
print "Lengths: mean", np.mean(lengths), "std", np.std(lengths)
print "Rewards: mean", np.mean(rewards), "std", np.std(rewards)
# env.monitor.close()
