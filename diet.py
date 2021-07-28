import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import time
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os

# goal percentages original
total_calories = 2500 * 7
percentage_prot = 0.3
percentage_carb = 0.5
percentage_fat = 0.2

# goal percentages
# total_calories = 3000 * 7
# percentage_prot = 0.3
# percentage_carb = 0.5
# percentage_fat = 0.2
number_generation = 50
# total_calories = int(input('Entre com o total de calorias por dia: '))
# days_period = int(input('Entre com o periodo de dias: '))
# total_calories = total_calories * days_period

# percentage_prot = float(input('Entre com a porcentagem de proteínas em formato decimal: '))
# percentage_carb = float(input('Entre com a porcentagem de carboidratos em formato decimal: '))
# percentage_fat = float(input('Entre com a porcentagem de gordura em formato decimal: '))

id_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
dir = f"logs/run_{int(percentage_prot * 10)}_{int(percentage_carb * 10)}_{int(percentage_fat * 10)}_{id_time}"
os.mkdir(dir)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(dir, "run_{}_{}.txt".format(number_generation, id_time)), "w")

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

# goal grams
gram_prot = cal_prot / prot_cal_p_gram
gram_carb = cal_carb / carb_cal_p_gram
gram_fat = cal_fat / fat_cal_p_gram
# print(gram_prot, gram_carb, gram_fat)

# per week: min, max, cal unit, prot g,  fat g, carb g
products_table = pd.DataFrame.from_records([
    ['Banana 1u', 0, 4, 89, 1, 0, 23],
    ['Mandarin 1u', 0, 4, 40, 1, 0, 10],
    ['Ananas 100g', 0, 7, 50, 1, 0, 13],
    ['Grapes 100g', 0, 7, 76, 1, 0, 17],
    ['Chocolate 1 bar', 0, 4, 230, 3, 13, 25],

    ['Hard Cheese 100g', 0, 8, 350, 28, 26, 2],
    ['Soft Cheese 100g', 0, 8, 374, 18, 33, 1],
    ['Pesto 100g', 0, 8, 303, 3, 30, 4],
    ['Hoummous 100g', 0, 8, 306, 7, 25, 11],
    ['Aubergine Paste 100g', 0, 4, 228, 1, 20, 8],

    ['Protein Shake', 0, 5, 160, 30, 3, 5],
    ['Veggie Burger 1', 0, 5, 220, 21, 12, 3],
    ['Veggie Burger 2', 0, 12, 165, 16, 9, 2],
    ['Boiled Egg', 0, 8, 155, 13, 11, 1],
    ['Backed Egg', 0, 16, 196, 14, 15, 1],

    ['Baguette Bread Half', 0, 3, 274, 10, 0, 52],
    ['Square Bread 1 slice', 0, 3, 97, 3, 1, 17],
    ['Cheese Pizza 1u', 0, 3, 903, 36, 47, 81],
    ['Veggie Pizza 1u', 0, 3, 766, 26, 35, 85],

    ['Soy Milk 200ml', 0, 1, 115, 8, 4, 11],
    ['Soy Chocolate Milk 250ml', 0, 3, 160, 7, 6, 20],

])
products_table.columns = ['Name', 'Min', 'Max',
                          'Calories', 'Gram_Prot', 'Gram_Fat', 'Gram_Carb']

# print(products_table.to_csv(index=False))

# extract the information of products in a format that is easier to use in the deap algorithms cost function
cal_data = products_table[['Gram_Prot', 'Gram_Fat', 'Gram_Carb']]

prot_data = list(cal_data['Gram_Prot'])
fat_data = list(cal_data['Gram_Fat'])
carb_data = list(cal_data['Gram_Carb'])


# the random initialization of the genetic algorithm is done here
# it gives a list of integers with for each products the number of times it is bought
def n_per_product():
    return random.choices(range(0, 10), k=21)

 # this is the function used by the algorithm for evaluation
# I chose it to be the absolute difference of the number of calories in the planning and the goal of calories
def evaluate_mono(individual):
    individual = individual[0]
    tot_prot = sum(x*y for x, y in zip(prot_data, individual))
    tot_fat = sum(x*y for x, y in zip(fat_data, individual))
    tot_carb = sum(x*y for x, y in zip(carb_data, individual))
    cals = prot_cal_p_gram * tot_prot + carb_cal_p_gram * \
        tot_carb + fat_cal_p_gram * tot_fat
    return abs(cals - total_calories),


def evaluate_mult(individual):
    individual = individual[0]
    tot_prot = sum(x*y for x, y in zip(prot_data, individual))
    tot_fat = sum(x*y for x, y in zip(fat_data, individual))
    tot_carb = sum(x*y for x, y in zip(carb_data, individual))
    cals = prot_cal_p_gram * tot_prot + carb_cal_p_gram * \
        tot_carb + fat_cal_p_gram * tot_fat
    return abs(cals - total_calories), \
        abs(tot_prot - gram_prot), \
        abs(tot_fat - gram_fat), \
        abs(tot_carb - gram_carb), \

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
    while g < number_generation:
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

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        gen.loc[:,g] = fits
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
sum_err_mono = 0
sum_err_mult = 0

for i in range(35):
    print("-- Round %i --" % i)
    # this is the setup of the deap library: registering the different function into the toolbox
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

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

    # in this second version, we optimize for the four components of the shopping list: calories, protein, fat and carbs
    # if we need to make everything as important, we should add a weight to them
    # we know that there are 30% protein calories, 20% fat and 50% carbs.
    weights = (-1., -1. / 0.3, -1. / 0.2, -1./0.5)

    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)

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
    summary["univ_error"] = (
        summary["goal"] - summary["univariate"]).apply(abs)
    summary["multiv_error"] = (
        summary["goal"] - summary["multivariate"]).apply(abs)

    err_list.append(summary)

    # print(summary)

    sum_err_mono = (sum_err_mono + summary["univ_error"].sum()) / 2
    sum_err_mult = (sum_err_mult + summary["multiv_error"].sum()) / 2

    # print((summary["univ_error"].sum(), summary["multiv_error"].sum()))

    # Shopping list
    # print(products_table[['Name', 'multivariate_choice', 'univariate_choice']])
    shop_list.append(products_table[['Name', 'multivariate_choice', 'univariate_choice']])


# average values in 35 rounds
df_concat = pd.concat(gen_list_mono)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()

# plot boxplot
_, ax = plt.subplots()
columns = [i for i in range(1, number_generation, int(number_generation/10))]
columns.append(number_generation)
genFilteredColumns = df_means[columns]
ax.set_xlabel('Number generation')
ax.set_ylabel('Fitness')
boxplot = genFilteredColumns.boxplot(ax=ax, grid=False)
plt.savefig(os.path.join(dir, f"boxplot_{number_generation}_{id_time}.png"))

# plot line chart
_, ax = plt.subplots()
mean = pd.DataFrame(columns=['mean'])
for i in range(1,number_generation+1):
    mean.loc[i] = df_means[i].mean()
ax.set_xlabel('Number generation')
ax.set_ylabel('Fitness')
mean.plot(ax=ax)

plt.savefig(os.path.join(dir, f"mean_{number_generation}_{id_time}.png"))

# average errors in 35 rounds
df_concat = pd.concat(err_list)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()
df_means.to_csv(os.path.join(dir, f'errors_{number_generation}_{id_time}.csv'))

# df_means = df_means.drop(df_means.index[[0]])
# # plot pie chart
# _, ax = plt.subplots()
# ax.pie(df_means['univariate'].to_list(), labels=df_means.index.to_list(), autopct='%1.1f%%', startangle=90)
# ax.axis('equal')
# plt.savefig(os.path.join(dir, f"univariate{number_generation}_{id_time}.png"))

# # plot pie chart
# _, ax = plt.subplots()
# ax.pie(df_means['multivariate'].to_list(), labels=df_means.index.to_list(), autopct='%1.1f%%', startangle=90)
# ax.axis('equal')
# plt.savefig(os.path.join(dir, f"multivariate{number_generation}_{id_time}.png"))

# average shopping lists in 35 rounds
df_concat = pd.concat(shop_list)
by_row_index = df_concat.groupby(df_concat.index)
df_means2 = by_row_index.mean()
df_means2['multivariate_choice'] = df_means2['multivariate_choice'].round()
df_means2['univariate_choice'] = df_means2['univariate_choice'].round()
df_means2.to_csv(os.path.join(dir, f'shop_{number_generation}_{id_time}.csv'))


print("mono execution time = {}".format(mono_time))
print("mult execution time = {}".format(mult_time))
print("mono error = {}".format(sum_err_mono))
print("mult error = {}".format(sum_err_mult))
