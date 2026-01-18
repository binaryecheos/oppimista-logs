import itertools

def main():
    letters = ['b', 'o', 'b', 'a']
    perms = set(itertools.permutations(letters))
    for perm in perms:
        pretty_perm = "".join(perm)
        print(pretty_perm)

if __name__ == "__main__":
    main()

import math

def main():
    n = math.factorial(4) 
    d = math.factorial(2)
    print(n / d)



if __name__ == "__main__":
    main()