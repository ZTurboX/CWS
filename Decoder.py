import copy
from Feature import Feature


class Decoder:
    def __init__(self, beam_size, get_score):
        self.beam_size = beam_size
        self.get_score = get_score

    def beamSearch(self, sent):
        src = [([], 0.0)]
        tgt = []
        split_sent = [char for char in sent if(char != ' ') and (char != '\n')]
        for index in range(len(split_sent)):
            char = split_sent[index]

            for item, score in src:
                item1 = copy.deepcopy(item)
                new_sent = item1 + [char]

                tgt.append((new_sent, score + self.get_score(new_sent)))
                if len(item) > 0:
                    item2 = item.copy()
                    item2[-1] += char

                    tgt.append(
                        (item2, score - self.get_score(item) + self.get_score(item2)))

            src = self.get_best_item(tgt, self.beam_size)
            tgt = []

        return self.get_best_item(src, 1)[0][0]

    def get_best_item(self, item, beam_size):
        return sorted(item, key=lambda x: x[1], reverse=True)[0:beam_size]
