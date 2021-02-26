import json
import matplotlib.pyplot as plt

def load_data(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def plot_cost(data):

    for d in data:
        costs = d["result"]["costs"]
        steps = []
        values = []
        for step, value in costs.items():
            steps.append(int(step))
            values.append(value)
    
        plt.plot(steps, values)
    
    plt.show()


if __name__ == "__main__":
    files = ["logs/W5_L20_DOfalse.json",
             "logs/W5_L20_DOtrue.json"]

    data = []

    for f in files:
        data.append(load_data(f))

    plot_cost(data)
