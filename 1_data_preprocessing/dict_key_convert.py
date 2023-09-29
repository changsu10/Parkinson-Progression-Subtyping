"""

Convert key type of dictionary.

"""

def str2int(input_dict):
    out_dict = {}
    for k in input_dict:
        try:
            new_k = int(k)
            out_dict[new_k] = input_dict[k]
        except:
            print("Key %s is not a integer." % k)
    return out_dict
if __name__ == "__main__":
    test1 = {"3401":1}
    test2 = {"abcd":2, "2333":1}
    print(str2int(test1))
    print(str2int(test2))