import json
import matplotlib.pyplot as plt


def load_data(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def plot_cost_stupid(data):
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
            steps = [s*3 for s in steps]

        plt.plot(steps, values)
    
    plt.show()


if __name__ == "__main__":
    files = ["logs/W5_L10_DOtrue.json",
             "logs/W5_L10_DOfalse.json"]

    data = []

    for f in files:
        data.append(load_data(f))

    plot_cost_stupid(data)
