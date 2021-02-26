import json
import matplotlib.pyplot as plt


def load_data(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def plot_cost(data):

    for d in data:
        costs = d["result"]["costs"]
        final_costs = d["result"]["final_cost"]
        steps = []
        values = []
        for step, value in costs.items():
            steps.append(int(step))
            values.append(value)
        steps.append(steps[-1]+5)
        values.append(final_costs)

    
        plt.plot(steps, values)
    
    plt.show()


if __name__ == "__main__":
    files = ["logs/W5_L20_DOfalse.json",
             "logs/W5_L20_DOtrue.json"]

    data = []

    for f in files:
        data.append(load_data(f))

    plot_cost(data)
