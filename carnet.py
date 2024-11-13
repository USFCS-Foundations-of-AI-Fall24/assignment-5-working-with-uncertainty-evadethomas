from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Key", "Starts"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "Key"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "Key": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_key = TabularCPD(
    variable="Key", varible_card=2,
    values=[[0.7], [0.3]],
    evidence_card=[2],
    state_names={"key": ["yes", "no"]},
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

print("\nGiven that the car will not move, the probability that the battery is not working:\n")

print(car_infer.query(variables=["Battery"],evidence={"Moves": "no"}))

print("\nGiven that the radio is not working, the probability that the car will not start:\n")

print(car_infer.query(variables=["Starts"],evidence={"Radio": "Doesn't turn on"}))

print("\nGiven that the battery is working, does the probability of the radio working change if we discover that the car has gas in it:")

print("After discovering that there is gas: \n")

print(car_infer.query(
    variables=["Radio"],
    evidence={"Battery": "Works", "Gas": "Full"},
))

print("\nGiven that the car doesn't move, how does the probability of the ignition failing change if we observe that\n the car dies not have gas in it?\n")

print("Ignition failing without a gas level observation: \n")
print(car_infer.query(
    variables=["Ignition"],
    evidence={"Moves": "no"},
))

print("Ignition failing with a gas level observation of empty: \n")
print(car_infer.query(
    variables=["Ignition"],
    evidence={"Moves": "no", "Gas":"Empty"},
))

print("What is the probability that the car starts if the radio works and it has gas in it?")

print(car_infer.query(
    variables=["Starts"],
    evidence={"Radio": "turns on", "Gas":"Full"},
))

print("Showing the probability that the key is not present given that the car does not move.")

print(car_infer.query(
    variables=["Key"],
    evidence={"Moves":"no"},
))