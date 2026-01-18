import itertools
import math

VALUES = '23456789TJQKA'
SUITS = 'CDHS'

def main():
    cards = []
    for value in VALUES:
        for suit in SUITS:
            cards.append(value + suit)
    subsets = set(itertools.combinations(cards, 5))
    print(len(subsets))
    # for hands in subsets:
    #     print(hands)
    print(math.comb(52, 5))
    print(math.factorial(52) // (math.factorial(5) * math.factorial(52 - 5)))


if __name__ == "__main__":
    main()