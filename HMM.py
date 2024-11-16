from time import sleep

import numpy as np
import random
import argparse
import codecs
import os
import numpy
from sympy import sequence


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)



# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        emit_dict = {}
        trans_dict = {}

        file_types = ['.emit', '.trans']
        for i in range(0, len(file_types)):
            if file_types[i] == '.emit':
                file_dict = emit_dict
            else:
                file_dict = trans_dict
            with open(basename + file_types[i], 'r') as file:
                for line in file:
                    line_split = line.strip().split(" ")
                    if (line == "\n") :
                        continue
                    if line_split[0] not in file_dict:
                        file_dict[line_split[0]] = {}
                    file_dict[line_split[0]][line_split[1]] = line_split[2]

        self.emissions=emit_dict
        self.transitions=trans_dict

        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        trans_state_name = "#"
        trans = []
        emis = []

        for i in range(0, n):
            # Randomly select an outcome based on the probabilities
            trans_state = self.transitions[trans_state_name]
            trans_outcome = np.random.choice(list(trans_state.keys()), p=list(trans_state.values()))
            trans_state_name =  str(trans_outcome)
            trans.append(trans_state_name)
            emis_name = self.emissions[trans_state_name]
            emis_outcome = np.random.choice(list(emis_name.keys()), p=list(emis_name.values()))
            emis_outcome_string = str(emis_outcome)
            emis.append(emis_outcome_string)

        seq = Sequence(trans, emis)
        return seq

    def forward(self, sequence):
        observ_count = len(sequence.outputseq)
        state_count = len(self.transitions)
        matrix = np.zeros((observ_count, state_count))
        obs_1 = sequence.outputseq[0]
        initial_state = list(self.transitions.keys())[0]

        for s in range(1, state_count):
            state = list(self.transitions.keys())[s]
            try:
                probability1 = self.transitions[initial_state][state]
            except KeyError:
                probability1 = 0

            try:
                probability2 = self.transitions[initial_state][state]
            except KeyError:
                probability2 = 0

            matrix[0,s] = float(probability1) * float(probability2)

        for i in range(1, observ_count):
            for s in range(1, state_count):
                sum = 0
                state = list(self.transitions.keys())[s]
                for s2 in range(1, state_count):
                    state_2 = list(self.transitions.keys())[s2]

                    try:
                        probability1 = float(self.transitions[state_2][state])
                    except KeyError:
                        probability1 = 0

                    try:
                        probability2 = float(self.emissions[state_2][sequence.outputseq[i]])
                    except KeyError:
                        probability2 = 0


                    sum += matrix[i - 1, s2] * probability1 * probability2
                    matrix[i,s2] = sum
        most_likely = list(self.transitions.keys())[np.argmax(matrix[-1])]
        return most_likely


    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.


    def viterbi(self, sequence):
        pass



        #M = np.zeros((t, s))


    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

def main():
    parser = argparse.ArgumentParser(description="Arguments for HMM")
    parser.add_argument('file_name', type=str, help='enter the data to process')
    parser.add_argument('--generate', type=int, help='number of observations requested', default = 0)
    parser.add_argument('--forward', type=str, default=None)


    args = parser.parse_args()

    h = HMM()
    h.load(args.file_name)

    if args.forward == None:
        print(str(h.generate(args.generate)))
    else:
        seq = h.generate(args.generate)
        print(h.forward(seq))

    h2 = HMM()
    h2.load("lander")
    lander_result = h2.forward(h2.generate(20))
    print("Lander running...")
    sleep(2)


    if lander_result != '4,3' and lander_result != '3,4' and lander_result != '4,4' and lander_result != '5,5' and lander_result != '2,5' :
        print("Safe to land!")
    else:
        print("Not safe to land!")

if __name__ == "__main__":
    main()




