import collections

class Alphabet(object):
    def __init__(self, alphabet_file):
        self.alphabet = []
        with open(alphabet_file) as f:
            self.alphabet = [""] + list(f.read().replace("\n", "")) + ["#UNK#"]
        self.alphabet_dict = {char: i for i, char in enumerate(self.alphabet)}
        
    def __len__(self):
        return len(self.alphabet)

    def __getitem__(self, index):
        return self.alphabet[index]

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            list [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            list [n]: length of each text.
        """
        if isinstance(text, str):
            text = [self.alphabet_dict.get(char, self.alphabet_dict["#UNK#"]) for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (text, length)

    def decode(self, text, length=None, raw=False):
        """Decode encoded texts back into strs.

        Args:
            list [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            list [n]: length of each text, leave None if only have one text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length is None:
            length = len(text)
            if raw:
                return ''.join([self.alphabet[i] for i in text])
            else:
                char_list = []
                for i in range(length):
                    if text[i] != 0 and (not (i > 0 and text[i - 1] == text[i])):
                        char_list.append(self.alphabet[text[i]])
                return ''.join(char_list)
        else:
            assert len(text) == sum(length), \
                "texts with length: {} does not match declared length: {}".format(len(text), sum(length))

            texts = []
            index = 0
            for i in range(len(length)):
                l = length[i]
                texts.append(self.decode(text[index:index + l], raw=raw))
                index += l
            return texts


if __name__ == "__main__":
    alphabet = Alphabet("alphabet.txt")
    print(len(alphabet), alphabet[1000], alphabet[-1])
    print(alphabet.encode(["hello,world", "测试", "耑釆岀"]))
    print(alphabet.decode(
        [72, 69, 76, 76, 79, 12, 87, 79, 82, 76, 68, 1063, 3305, len(alphabet) - 1, len(alphabet) - 1, len(alphabet) - 1],
        [11, 2, 3]
    ))
