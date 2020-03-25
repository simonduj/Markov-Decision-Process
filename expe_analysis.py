import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

expe_circle = [None] * 20
for i in range(0,20):
	expe_circle[i] = pd.read_pickle("experiments/circle/layout_" + str(i) + "_expe.pkl")
	expe_circle[i].columns = ['dice1', 'dice2', 'random', 'markov', 'rand. markov']

expe_no_circle = [None] * 20
for i in range(0,20):
	expe_no_circle[i] = pd.read_pickle("experiments/no_circle/layout_" + str(i) + "_expe.pkl")
	expe_no_circle[i].columns = ['dice1', 'dice2', 'random', 'markov', 'rand. markov']


expe_circle_total = pd.concat(expe_circle)
expe_no_circle_total = pd.concat(expe_no_circle)


boxplot_circle = expe_circle_total.boxplot(showfliers=False)
plt.xlabel('Empirical cost')
plt.title('Empirical cost distribution for each strategy (circle)')

plt.savefig('experiments/graphs/box_circle.png')
plt.show()

boxplot_no_cirle = expe_no_circle_total.boxplot(showfliers=False)
plt.xlabel('Empirical cost')
plt.title('Empirical cost distribution for each strategy (no circle)')

plt.savefig('experiments/graphs/box_no_cirle.png')
plt.show()


###################################################################
#Analysis with respect to the number of traps
#CIRCLE
expe_circle_trap = [None] * 10
for i in range(0,19, 2):
	expe_circle_trap[int(i/2)] = pd.concat([expe_circle[i], expe_circle[i+1]])

x_axis = [1,2,3,4,5,6,7,8,9,10]
#get points for a line
dice1 = [data['dice1'].mean() for data in expe_circle_trap]
dice2 = [data['dice2'].mean() for data in expe_circle_trap]
random = [data['random'].mean() for data in expe_circle_trap]
markov = [data['markov'].mean() for data in expe_circle_trap]
rand_markov = [data['rand. markov'].mean() for data in expe_circle_trap]

y_axis = [dice1, dice2, random, markov, rand_markov]
y_names = ['dice1']

for i in range(0, len(y_axis)):
	plt.plot(x_axis, y_axis[i], label=expe_circle[i].columns[i])
plt.xlabel("Number of traps")
plt.ylabel("Mean empirical cost")
plt.legend()
plt.title('Mean empirical cost vs the number of traps (circle)')
plt.savefig('experiments/graphs/vs_trap_circle.png')
plt.show()

########################################################################
expe_no_circle_trap = [None] * 10
for i in range(0,19, 2):
	expe_no_circle_trap[int(i/2)] = pd.concat([expe_no_circle[i], expe_no_circle[i+1]])

x_axis = [1,2,3,4,5,6,7,8,9,10]
#get points for a line
dice1 = [data['dice1'].mean() for data in expe_no_circle_trap]
dice2 = [data['dice2'].mean() for data in expe_no_circle_trap]
random = [data['random'].mean() for data in expe_no_circle_trap]
markov = [data['markov'].mean() for data in expe_no_circle_trap]
rand_markov = [data['rand. markov'].mean() for data in expe_no_circle_trap]

y_axis = [dice1, dice2, random, markov, rand_markov]

for i in range(0, len(y_axis)):
	plt.plot(x_axis, y_axis[i], label=expe_no_circle[i].columns[i])
plt.xlabel("Number of traps")
plt.ylabel("Mean empirical cost")
plt.legend()
plt.title('Mean empirical cost vs the number of traps (no circle)')
plt.savefig('experiments/graphs/vs_trap_no_circle.png')
plt.show()


##########################################
#get the real costs
mean_real_cost_circle = [None] * 20

for i in range(0,20):
	mean_real_cost_circle[i] = expe_circle[i]['markov'].mean()

mean_real_cost_no_circle = [None] * 20
for i in range(0,20):
	mean_real_cost_no_circle[i] = expe_no_circle[i]['markov'].mean()

print(mean_real_cost_circle)
print(mean_real_cost_no_circle)




