from deap import base, creator, tools

def generate_individual(num_repeat):
    a = 'CCTTGGAA'
    b=''
    for i in range(num_repeat):
        b = b+a
    return b

IND_SIZE=100
toolbox = base.Toolbox()
toolbox.register('Individual', generate_individual)
indi = toolbox.Individual(IND_SIZE)
print(indi)



