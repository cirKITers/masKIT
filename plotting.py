import json
import matplotlib.pyplot as plt


def load_data(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def plot_cost_stupid(data):
    data_length = 0

    for d in data:
        costs = d["result"]["costs"]
        if data_length == 0:
            data_length = len(costs)
        final_costs = d["result"]["final_cost"]
        steps = []
        values = []
        for step in list(costs.keys())[:data_length]:
            steps.append(int(step))
            values.append(costs[step])
        steps.append(steps[-1]+5)
        values.append(final_costs)
    
        plt.plot(steps, values)
    
    plt.show()


if __name__ == "__main__":
    files = ["logs/W5_L20_DOtrue.json",
             "logs/W5_L20_DOfalse.json"]

    data = []

    for f in files:
        data.append(load_data(f))

    plot_cost_stupid(data)
