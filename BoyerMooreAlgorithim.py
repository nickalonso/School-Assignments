# Boyer Moore Algorithim
# This is an assortment of code I found off of 3 different websites

# http://wiki.dreamrunner.org/public_html/Algorithms/TheoryOfAlgorithms/Boyer-Moore_string_search_algorithm.html
# https://gist.github.com/mengzhuo/d713cb8be6c871995b79
# https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/

def bad_character_table(pattern):
    # copied from ascii table, create a list of every possible character (a..,A..,1...,./")

    alphabet = ['0','1','2','3','4','5','6','7','8','9',':',';','<','>','=','?','@','A','B','C','D','E','F','G',
                'H','I','J','K','L','M','P','Q','R','S','T','Q','V','W','X','Y','Z','[',']','/', '^','_','!','#',
                '$','%','&','(',')','*','+',',','-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                'q','r','s','t','u','v','w','x','y','z',' ', '.']
    length = len(pattern)
    table = {}

    # initialize all alphabetCharacter:<values> to length
    for char in alphabet:
        table[char] = length

    # loop through pattern
    for i, c in enumerate(pattern):
        if i == length-1 and not c in table:
            table[c] = length
        else:
            table[c] = length - i - 1
    return table

def match_length(S, idx1, idx2):
    # Returns the length of the match of the substrings of S beginning at idx1 and idx2.

    if idx1 == idx2:
        return len(S) - idx1
    match_count = 0
    while idx1 < len(S) and idx2 < len(S) and S[idx1] == S[idx2]:
        match_count += 1
        idx1 += 1
        idx2 += 1
    return match_count

def fundamental_preprocess(S):
    # Z[i] is the length of the substring beginning at i which is also a prefix of S.
    if len(S) == 0: # Handles case of empty string
        return []
    if len(S) == 1: # Handles case of single-character string
        return [1]
    z = [0 for x in S]
    z[0] = len(S)
    z[1] = match_length(S, 0, 1)
    for i in range(2, 1+z[1]): # Optimization from exercise 1-5
        z[i] = z[1]-i+1
    # Defines lower and upper limits of z-box
    l = 0
    r = 0
    for i in range(2+z[1], len(S)):
        if i <= r: # i falls within existing z-box
            k = i-l
            b = z[k]
            a = r-i+1
            if b < a: # b ends within existing z-box
                z[i] = b
            else: # b ends at or after the end of the z-box, we need to do an explicit match to the right of the z-box
                z[i] = a+match_length(S, a, r+1)
                l = i
                r = i+z[i]-1
        else: # i does not reside within existing z-box
            z[i] = match_length(S, 0, i)
            if z[i] > 0:
                l = i
                r = i+z[i]-1
    return z

def good_suffix_table(S): # input S should be the pattern
    """"
    Generates L for S, an array used in the implementation of the strong good suffix rule.
    L[i] = k, the largest position in S such that S[i:] (the suffix of S starting at i) matches
    a suffix of S[:k] (a substring in S ending at k). Used in Boyer-Moore, L gives an amount to
    shift P relative to T such that no instances of P in T are skipped and a suffix of P[:L[i]]
    matches the substring of T matched by a suffix of P in the previous match attempt.
    Specifically, if the mismatch took place at position i-1 in P, the shift magnitude is given
    by the equation len(P) - L[i]. In the case that L[i] = -1, the full shift table is used.
    Since only proper suffixes matter, L[0] = -1.
    """""
    L = [-1 for c in S]
    N = fundamental_preprocess(S[::-1]) # S[::-1] reverses S
    N.reverse()
    for j in range(0, len(S)-1):
        i = len(S) - N[j]
        if i != len(S):
            L[i] = j
    return L

def full_shift_table(S):
    """"
    Generates F for S, an array used in a special case of the good suffix rule in the Boyer-Moore
    string search algorithm. F[i] is the length of the longest suffix of S[i:] that is also a
    prefix of S. In the cases it is used, the shift magnitude of the pattern P relative to the
    text T is len(P) - F[i] for a mismatch occurring at i-1.
    """""
    F = [0 for c in S]
    Z = fundamental_preprocess(S)
    longest = 0
    for i, zv in enumerate(reversed(Z)):
        longest = max(zv, longest) if zv == i+1 else longest
        F[-i-1] = longest

    return F

def boyer_moore(pattern, text):

    R = bad_character_table(pattern)
    L = good_suffix_table(pattern)  # store good suffix table
    F = full_shift_table(pattern)
    pattern_length = len(pattern)
    text_length = len(text)
    if pattern_length > text_length:
        return -1

    table = bad_character_table(pattern)
    index = pattern_length - 1
    pattern_index = pattern_length - 1
    print("\nText: ", text)
    print("Pattern: ", pattern, "\n")
    while index < text_length:
        # for every shift, initialize number of matches to 0
        numberOfMatches = 0
        if pattern[pattern_index] == text[index]: # do have a match for aligned characters
            numberOfMatches += 1
            if pattern_index == 0:
            # reached the beginning of the pattern, the whole pattern matched
            # no need to shift, return the current index
                return index

            else: # have not reached the beginning of the pattern
                pattern_index -= 1
                index -= 1

        else: # no match
            """get max between goodsuffix table (index with "number of matches") and 
            badcharshift table (index with mismatched character)"""
            print("current shift at index =",index)
            print("table value of index:",table[text[index]])
            badcharshift = max([R[text[index]] - numberOfMatches, 1])
            if numberOfMatches == 0:  # numberofmatches = k, if zero use bad char table
                shift_size = badcharshift # translated from max([t(c) - k,1]) from text
            elif numberOfMatches > 0: # if greater than 0, use goodsuffix table
                goodsuffixshift = L[numberOfMatches]
                shift_size = max(badcharshift, goodsuffixshift)
            index += shift_size
            print("new index: ", index, "\n")
            print("Suffix: ", text[index:(index+len(pattern))])

    return -1

# Test case for the above function
def main():
    text = "Hello Dr. Ghosh, how are you?"
    pattern = "how"
    match_position = boyer_moore(pattern, text)
    print("Match found at index: ", match_position)

if __name__ == '__main__':
    main()

