import random


def word_repetition(batch_text, dup_rate=0.2):
    dst_text = []

    for text in batch_text:
        text_list = text.split(' ')
        actual_len = len(text_list)
        if actual_len > 0:
            dup_len = random.randrange(min(10, actual_len, max(2, int(dup_rate * actual_len))))
            dup_word_index = sorted(random.sample(range(0, actual_len - 1), dup_len))

            for i in dup_word_index:
                text_list[i] = text_list[i] + ' ' + text_list[i]
            dst_text.append(' '.join(text_list))
        else:
            dst_text.append(text)

    return dst_text
