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

        if d["call"][2]["use_dropout"]:
            steps = [s * 3 for s in steps]

        plt.plot(steps, values)

    plt.show()


def plot_cost_stupid(data):
    data_length = 0
    for d in data:
        costs = d["result"]["costs"]
        if data_length == 0:
            data_length = len(costs)
        steps = []
        values = []
        for step in list(costs.keys())[:data_length]:
            steps.append(int(step))
            values.append(costs[step])

        plt.plot(steps, values)
    
    plt.show()


def plot_cost_normalized(data):
    real_data_length = 0

    for d in data:
        costs = d["result"]["costs"]
        branches = d["result"]["branches"]
        steps = []
        values = []
        step_counter = 0
        if real_data_length == 0:
            real_data_length += len(costs) * 5
            real_data_length += len(branches) * 3

        for i in range(real_data_length):
            key = f"{i}"
            if key in branches:
                step_counter += 3
            if key in costs:
                steps.append(i + step_counter)
                values.append(costs[key])

        label = "With dropout" if d["call"][2]["use_dropout"] else "Without dropout"
        plt.plot(steps, values, label=label)

    plt.rcParams.update({'font.size': 15})
    plt.legend(loc="upper right")
    plt.title(label="Wires: {}, Layers: {}".format(d["call"][2]["wires"], d["call"][2]["layers"]), fontdict={"fontsize": 18})
    plt.xlabel('Training steps', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.show()


if __name__ == "__main__":
    files = ["logs/W5_L10_DOtrue.json",
             "logs/W5_L10_DOfalse.json"]

    data = []

    for f in files:
        data.append(load_data(f))

    # plot_cost(data)
    # plot_cost_stupid(data)
    plot_cost_normalized(data)
