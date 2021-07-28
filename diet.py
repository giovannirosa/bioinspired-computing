import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import time
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

ERROR_METHOD = 'rmse'
# ERROR_METHOD = 'mae'

NUMBER_GEN = 50
ROUNDS = 35

# goal percentages
total_calories = 2500 * 7
percentage_prot = 0.3
percentage_carb = 0.5
percentage_fat = 0.2

# total_calories = int(input('Entre com o total de calorias por dia: '))
# days_period = int(input('Entre com o periodo de dias: '))
# total_calories = total_calories * days_period

# percentage_prot = float(input('Entre com a porcentagem de proteínas em formato decimal: '))
# percentage_carb = float(input('Entre com a porcentagem de carboidratos em formato decimal: '))
# percentage_fat = float(input('Entre com a porcentagem de gordura em formato decimal: '))

id_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
dir = f"logs/run_{ERROR_METHOD}_{NUMBER_GEN}_{id_time}"
os.mkdir(dir)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(
            dir, f"run_{ERROR_METHOD}_{NUMBER_GEN}_{id_time}.txt"), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()

# compute total calories per macro
cal_prot = round(percentage_prot * total_calories)
cal_carb = round(percentage_carb * total_calories)
cal_fat = round(percentage_fat * total_calories)
# print(cal_prot, cal_carb, cal_fat)

# fixed info on macro nutriments: calories per gram of protein, carb and fat
prot_cal_p_gram = 4
carb_cal_p_gram = 4
fat_cal_p_gram = 9

# per week: min, max, cal unit, prot g,  fat g, carb g
products_table = pd.DataFrame.from_records([
    ['Banana 1u', 89, 1, 0, 23],
    ['Mandarin 1u', 40, 1, 0, 10],
    ['Ananas 100g', 50, 1, 0, 13],
    ['Grapes 100g', 76, 1, 0, 17],
    ['Chocolate 1 bar', 230, 3, 13, 25],

    ['Hard Cheese 100g', 350, 28, 26, 2],
    ['Soft Cheese 100g', 374, 18, 33, 1],
    ['Pesto 100g', 303, 3, 30, 4],
    ['Hoummous 100g', 306, 7, 25, 11],
    ['Aubergine Paste 100g', 228, 1, 20, 8],

    ['Protein Shake', 160, 30, 3, 5],
    ['Veggie Burger 1', 220, 21, 12, 3],
    ['Veggie Burger 2', 165, 16, 9, 2],
    ['Boiled Egg', 155, 13, 11, 1],
    ['Backed Egg', 196, 14, 15, 1],

    ['Baguette Bread Half', 274, 10, 0, 52],
    ['Square Bread 1 slice', 97, 3, 1, 17],
    ['Cheese Pizza 1u', 903, 36, 47, 81],
    ['Veggie Pizza 1u', 766, 26, 35, 85],

    ['Soy Milk 200ml', 115, 8, 4, 11],
    ['Soy Chocolate Milk 250ml', 160, 7, 6, 20],

])
products_table.columns = ['Name', 'Calories', 'Gram_Prot', 'Gram_Fat', 'Gram_Carb']

# print(products_table.to_csv(index=False))

# extract the information of products in a format that is easier to use in the deap algorithms cost function
cal_data = products_table[['Gram_Prot', 'Gram_Fat', 'Gram_Carb']]

prot_data = list(cal_data['Gram_Prot'])
fat_data = list(cal_data['Gram_Fat'])
carb_data = list(cal_data['Gram_Carb'])

limits_data = products_table[['Min', 'Max']]
min_data = list(limits_data['Min'])
max_data = list(limits_data['Max'])


# the random initialization of the genetic algorithm is done here
# it gives a list of integers with for each products the number of times it is bought
def n_per_product():
    return random.choices(range(0, 10), k=len(products_table.index))


 # this is the function used by the algorithm for evaluation
# I chose it to be the absolute difference of the number of calories in the planning and the goal of calories
def evaluate_mono(individual):
    individual = individual[0]
    tot_prot = sum(x*y for x, y in zip(prot_data,
                   individual)) * prot_cal_p_gram
    tot_fat = sum(x*y for x, y in zip(fat_data, 
                   individual)) * fat_cal_p_gram
    tot_carb = sum(x*y for x, y in zip(carb_data,
                   individual)) * carb_cal_p_gram
    cals = tot_prot + tot_carb + tot_fat
    return abs(cals - total_calories),


targets = [total_calories, cal_prot, cal_fat, cal_carb]


def evaluate_mult(individual):
    individual = individual[0]
    tot_prot = sum(x*y for x, y in zip(prot_data,
                   individual)) * prot_cal_p_gram
    tot_fat = sum(x*y for x, y in zip(fat_data, individual)) * fat_cal_p_gram
    tot_carb = sum(x*y for x, y in zip(carb_data,
                   individual)) * carb_cal_p_gram
    cals = tot_prot + tot_carb + tot_fat
    prediction = [cals, tot_prot, tot_fat, tot_carb]
    if ERROR_METHOD == 'rmse':
        rmse = np.sqrt(
            (sum((i - j)**2 for i, j in zip(prediction, targets)) / len(targets)))
        return rmse,
    else:
        mae = (abs(cals - total_calories) +
               abs(tot_prot - cal_prot) +
               abs(tot_fat - cal_fat) +
               abs(tot_carb - cal_carb)) / 4
        return mae,


mult_time = 0
mono_time = 0

# this is the definition of the total genetic algorithm is executed, it is almost literally copied from the deap library
def main(toolbox, multi=False):
    start_time = time.time()
    pop = toolbox.population(n=300)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of
    # fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < NUMBER_GEN:
        # A new generation
        g += 1
        # print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x*x for x in fits)
        # std = abs(sum2 / length - mean**2)**0.5
        gen.loc[:, g] = fits
        # print(min(fits), max(fits), mean, std)

    best = pop[np.argmin([toolbox.evaluate(x) for x in pop])]
    end_time = time.time()

    time_passed = end_time - start_time

    print(f"{time_passed} seconds")

    if multi:
        global mult_time
        mult_time = (mult_time + time_passed) / 2
    else:
        global mono_time
        mono_time = (mono_time + time_passed) / 2

    return best


gen_list_mono = []
gen_list_mult = []
err_list = []
shop_list = []

# this is the setup of the deap library: registering the different function into the toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

for i in range(ROUNDS):
    print("-- Round %i --" % i)

    toolbox = base.Toolbox()

    toolbox.register("n_per_product", n_per_product)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.n_per_product, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_mono)

    gen = pd.DataFrame()

    # as an example, this is what a population of 10 shopping lists looks like
    # print(toolbox.population(n=10))
    best_solution = main(toolbox)

    gen_list_mono.append(gen)

    products_table['univariate_choice'] = pd.Series(best_solution[0])
    # print(products_table[['Name', 'univariate_choice']])

    # ------------------------------------------------------------------------------------------------------------------

    # creator.create("FitnessMin", base.Fitness, weights=(-1.,))
    # creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("n_per_product", n_per_product)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.n_per_product, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_mult)

    gen = pd.DataFrame()

    best_solution = main(toolbox, True)

    gen_list_mult.append(gen)

    products_table['multivariate_choice'] = pd.Series(best_solution[0])

    products_table['univariate_gr_prot'] = products_table['univariate_choice'] * \
        products_table['Gram_Prot']
    products_table['univariate_gr_fat'] = products_table['univariate_choice'] * \
        products_table['Gram_Fat']
    products_table['univariate_gr_carb'] = products_table['univariate_choice'] * \
        products_table['Gram_Carb']
    products_table['univariate_cal'] = products_table['univariate_choice'] * \
        products_table['Calories']

    products_table['multivariate_gr_prot'] = products_table['multivariate_choice'] * \
        products_table['Gram_Prot']
    products_table['multivariate_gr_fat'] = products_table['multivariate_choice'] * \
        products_table['Gram_Fat']
    products_table['multivariate_gr_carb'] = products_table['multivariate_choice'] * \
        products_table['Gram_Carb']
    products_table['multivariate_cal'] = products_table['multivariate_choice'] * \
        products_table['Calories']

    summary = pd.DataFrame.from_records(
        [
            [products_table['univariate_gr_prot'].sum(
            ) * prot_cal_p_gram, products_table['multivariate_gr_prot'].sum() * prot_cal_p_gram, cal_prot],
            [products_table['univariate_gr_fat'].sum(
            ) * fat_cal_p_gram, products_table['multivariate_gr_fat'].sum() * fat_cal_p_gram, cal_fat],
            [products_table['univariate_gr_carb'].sum(
            ) * carb_cal_p_gram, products_table['multivariate_gr_carb'].sum() * carb_cal_p_gram, cal_carb],
            [products_table['univariate_cal'].sum(), products_table['multivariate_cal'].sum(),
             sum((cal_prot, cal_carb, cal_fat))]
        ])
    summary.columns = ['univariate', 'multivariate', 'goal']
    summary.index = ['prot', 'fat', 'carb', 'cal']

    err_list.append(summary)

    # Shopping list
    shop_list.append(
        products_table[['Name', 'multivariate_choice', 'univariate_choice']])


# average values in 35 rounds
df_concat = pd.concat(gen_list_mono)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()

# plot boxplot
_, ax = plt.subplots()
columns = [i for i in range(1, NUMBER_GEN, int(NUMBER_GEN/10))]
columns.append(NUMBER_GEN)
genFilteredColumns = df_means[columns]
ax.set_xlabel('Number generation')
ax.set_ylabel('Fitness')
boxplot = genFilteredColumns.boxplot(ax=ax, grid=False)
plt.savefig(os.path.join(dir, f"boxplot_{NUMBER_GEN}_{id_time}.png"))

# plot line chart
_, ax = plt.subplots()
mean = pd.DataFrame(columns=['mean'])
for i in range(1, NUMBER_GEN+1):
    mean.loc[i] = df_means[i].mean()
ax.set_xlabel('Number generation')
ax.set_ylabel('Fitness')
mean.plot(ax=ax)

plt.savefig(os.path.join(dir, f"mean_{NUMBER_GEN}_{id_time}.png"))

# average errors in 35 rounds
df_concat = pd.concat(err_list)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()
df_means = df_means.round()
df_means["univ_error"] = (
    df_means["goal"] - df_means["univariate"]).apply(abs)
df_means["multiv_error"] = (
    df_means["goal"] - df_means["multivariate"]).apply(abs)
df_means.to_csv(os.path.join(dir, f'errors_{NUMBER_GEN}_{id_time}.csv'))
sum_err_mono = df_means["univ_error"].sum()
sum_err_mult = df_means["multiv_error"].sum()

df_means = df_means.drop(df_means.index[[0]])
# plot pie chart
_, ax = plt.subplots()
ax.pie(df_means['univariate'].to_list(),
       labels=df_means.index.to_list(), autopct='%1.1f%%', startangle=90)
ax.axis('equal')
plt.savefig(os.path.join(dir, f"univariate_{NUMBER_GEN}_{id_time}.png"))

# plot pie chart
_, ax = plt.subplots()
ax.pie(df_means['multivariate'].to_list(),
       labels=df_means.index.to_list(), autopct='%1.1f%%', startangle=90)
ax.axis('equal')
plt.savefig(os.path.join(
    dir, f"multivariate_{NUMBER_GEN}_{id_time}.png"))

# average shopping lists in 35 rounds
df_concat = pd.concat(shop_list)
by_row_index = df_concat.groupby(df_concat.index)
df_means2 = by_row_index.mean()
df_means2['multivariate_choice'] = df_means2['multivariate_choice'].round()
df_means2['univariate_choice'] = df_means2['univariate_choice'].round()
df_means2.to_csv(os.path.join(dir, f'shop_{NUMBER_GEN}_{id_time}.csv'))


print("mono execution time = {}".format(mono_time))
print("mult execution time = {}".format(mult_time))
print("mono error = {}".format(sum_err_mono))
print("mult error = {}".format(sum_err_mult))
